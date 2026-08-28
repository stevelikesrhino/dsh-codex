/** OpenAI Codex adapter assembled from public dsh-llm-pi-ai extension points. */

import { access } from 'node:fs/promises'
import { homedir } from 'node:os'
import { resolve } from 'node:path'
import { createModels } from '@earendil-works/pi-ai'
import type { AuthContext, Context as PiContext, MutableModels, Provider, SimpleStreamOptions } from '@earendil-works/pi-ai'
import { openaiCodexProvider } from '@earendil-works/pi-ai/providers/openai-codex'
import { resolveRetryPolicy } from '@deepseek-ai/dsh-llm'
import type { GenerateOptions, PreparedAdapterCall, StreamChunk } from '@deepseek-ai/dsh-llm'
import { PiAiAdapter } from '@deepseek-ai/dsh-llm-pi-ai'
import type { ResolvedPiAiProviderProfile } from '@deepseek-ai/dsh-llm-pi-ai'
import type { AttachmentStore } from '@deepseek-ai/dsh-attachment'
import type { OpenAICodexCredentialStore } from './store.ts'
import { OPENAI_CODEX_PROVIDER } from './store.ts'
import { OpenAICodexResponseRuntime } from './responses.ts'
import type { ModelCatalogEntry, ResponseApiPreferences } from './tool-policy.ts'
import type { FastModeRegistry } from './fast-mode.ts'

/** Return a detached copy of the complete pi-ai Codex model catalog. */
export function openAICodexModelCatalog(): readonly ModelCatalogEntry[] {
  return openaiCodexProvider().getModels().map(model => ({ id: model.id, name: model.name }))
}

/** Provider idle ceiling used by the composite route. */
export const OPENAI_CODEX_STREAM_IDLE_TIMEOUT_MS = 300_000

/**
 * Image request budgets for the hand-built profile below. The profile bypasses
 * `resolveProfiles`, so these mirror the dsh-llm-pi-ai route defaults
 * (20 MiB request payload, 2048x2048 pixel budget, 1 MiB per encoded version)
 * that function would otherwise apply; the published package does not export
 * the constants.
 */
export const OPENAI_CODEX_MAX_REQUEST_IMAGE_BYTES = 20 * 1024 * 1024
export const OPENAI_CODEX_REQUEST_IMAGE_PIXEL_BUDGET = 2048 * 2048
export const OPENAI_CODEX_REQUEST_IMAGE_MAX_BYTES = 1024 * 1024

function record(value: unknown): Record<string, unknown> | undefined {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined
}

/** Lift the pre-rc.7 pi-ai replay shape into the current envelope on read. */
export function migrateLegacyOpenAICodexReplayState(value: unknown): unknown {
  const legacy = record(value)
  if (legacy?.['kind'] !== 'pi-ai' || legacy['version'] !== 1 || !Array.isArray(legacy['blocks'])) return value
  const {
    blocks,
    kind: _kind,
    version: _version,
    ...response
  } = legacy
  return {
    response: { ...response, kind: 'pi-ai', version: 2 },
    blocks,
  }
}

function migrateReplayHistory(options: GenerateOptions): GenerateOptions {
  let changed = false
  const messages = options.messages.map(message => {
    if (message.source.kind !== 'model' || message.source.replayState === undefined) return message
    const replayState = migrateLegacyOpenAICodexReplayState(message.source.replayState)
    if (replayState === message.source.replayState) return message
    changed = true
    return {
      ...message,
      source: { ...message.source, replayState },
    }
  })
  return changed ? { ...options, messages } : options
}

/**
 * Codex traffic rides on chatgpt.com, which is frequently reached through a
 * local proxy tunnel that blips for tens of seconds at a time. The dsh
 * default stops after 2 retries and caps scheduled delays at 10 seconds, so
 * this provider retries longer and backs off further to ride out such a blip.
 */
export const OPENAI_CODEX_RETRY_POLICY = resolveRetryPolicy({
  mode: 'normal',
  maxRetries: 5,
  backoff: { initialDelayMs: 1_000, maxDelayMs: 30_000, jitterRatio: 0.2 },
}, 'dsh-openai-codex retryPolicy')

/**
 * Give the generic dsh adapter a request-scoped bearer-token entry without
 * changing the provider's user-facing OAuth flow. The resolver accepts only
 * the explicit override supplied by this plugin; it never discovers an API
 * key from the environment or persistent api-key credentials.
 */
function isPayloadRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** Add the request-scoped Fast Mode hint without changing other payload fields. */
export function withOpenAICodexFastMode(
  provider: Provider,
  fastMode: FastModeRegistry | undefined,
): Provider {
  const streamSimple = provider.streamSimple
  return {
    ...provider,
    streamSimple(model, context: PiContext, options?: SimpleStreamOptions) {
      const enabled = provider.id === OPENAI_CODEX_PROVIDER
        && model.provider === OPENAI_CODEX_PROVIDER
        && fastMode?.isEnabled(options?.sessionId) === true
      if (!enabled) return streamSimple.call(provider, model, context, options)
      const previousOnPayload = options?.onPayload
      return streamSimple.call(provider, model, context, {
        ...options,
        async onPayload(payload, payloadModel) {
          const replaced = await previousOnPayload?.(payload, payloadModel)
          const nextPayload = replaced === undefined ? payload : replaced
          return isPayloadRecord(nextPayload)
            ? { ...nextPayload, service_tier: 'priority' }
            : nextPayload
        },
      })
    },
  }
}

function requestProvider(provider: Provider, fastMode?: FastModeRegistry): Provider {
  return {
    ...withOpenAICodexFastMode(provider, fastMode),
    auth: {
      ...provider.auth,
      apiKey: {
        name: 'OpenAI Codex OAuth bearer token',
        async resolve({ credential }) {
          const apiKey = credential?.key
          return apiKey === undefined || apiKey.length === 0
            ? undefined
            : { auth: { apiKey }, source: 'OAuth' }
        },
      },
    },
  }
}

/**
 * Preserve Harness call purpose through the pi-ai adapter: mark native
 * compaction calls and migrate legacy replay state on both dispatch entries
 * (`stream` for direct callers, `prepareCall` for the dsh-llm runtime).
 */
class OpenAICodexAdapter extends PiAiAdapter {
  constructor(
    options: ConstructorParameters<typeof PiAiAdapter>[0],
    private readonly responses: OpenAICodexResponseRuntime,
    private readonly visibleModelIds?: () => readonly string[],
  ) {
    super(options)
  }

  override async listModels(provider: string) {
    const models = await super.listModels(provider)
    const visibleModelIds = this.visibleModelIds?.()
    if (visibleModelIds === undefined) return models
    const visible = new Set(visibleModelIds)
    return models.filter(model => visible.has(model.id))
  }

  /**
   * Wrap one dispatch with the Codex call-scoped state: native compaction
   * marking for compaction purposes and the legacy replay-state migration.
   * dsh-llm dispatches through `prepareCall`, so both entry points share it.
   */
  private wrapDispatch(
    options: GenerateOptions,
    dispatch: (options: GenerateOptions) => AsyncIterable<StreamChunk>,
  ): AsyncIterable<StreamChunk> {
    const release = options.purpose === 'compaction'
      ? this.responses.enterCompaction(options.sessionId === undefined ? undefined : String(options.sessionId))
      : undefined
    return (async function* () {
      try {
        for await (const chunk of dispatch(migrateReplayHistory(options))) yield chunk
      } finally {
        release?.()
      }
    })()
  }

  override async *stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    yield* this.wrapDispatch(options, options => super.stream(options))
  }

  override prepareCall(provider: string, model: string, signal?: AbortSignal): Promise<PreparedAdapterCall> {
    return super.prepareCall(provider, model, signal).then(call => ({
      ...call,
      stream: (options: GenerateOptions) => this.wrapDispatch(options, options => call.stream(options)),
    }))
  }
}

/**
 * Ambient lookups for the collections this adapter builds: the process
 * environment and the host filesystem. The Codex route reaches them only when
 * the plugin's own store holds no credential; its OAuth flow owns auth
 * otherwise.
 */
const PROCESS_AUTH_CONTEXT: AuthContext = {
  env: async name => process.env[name],
  fileExists: async path => {
    const expanded = path === '~' || path.startsWith('~/')
      ? resolve(homedir(), path.slice(1).replace(/^\//, ''))
      : path
    try {
      await access(expanded)
      return true
    } catch {
      return false
    }
  },
}

/**
 * Create the Codex subscription adapter without requiring a dsh fork. The
 * public pi-ai adapter owns Harness message conversion, image attachment
 * resolution, streaming, and reasoning metadata. This plugin adds optional
 * Codex-native request state/compaction and supplies the provider OAuth token.
 */
export function createOpenAICodexAdapter(
  credentials: OpenAICodexCredentialStore,
  resolveAttachments: () => AttachmentStore | undefined,
  responsePreferences: () => ResponseApiPreferences,
  fastMode?: FastModeRegistry,
  visibleModelIds?: () => readonly string[],
): PiAiAdapter {
  const provider = openaiCodexProvider()
  const responses = new OpenAICodexResponseRuntime(responsePreferences)
  const profiles = new Map<string, ResolvedPiAiProviderProfile>([[OPENAI_CODEX_PROVIDER, {
    provider: OPENAI_CODEX_PROVIDER,
    displayName: 'OpenAI Codex',
    streamIdleTimeoutMs: OPENAI_CODEX_STREAM_IDLE_TIMEOUT_MS,
    retryPolicy: OPENAI_CODEX_RETRY_POLICY,
    configuredMaxTokens: new Map(),
    maxRequestImageBytes: OPENAI_CODEX_MAX_REQUEST_IMAGE_BYTES,
    requestImagePixelBudget: OPENAI_CODEX_REQUEST_IMAGE_PIXEL_BUDGET,
    requestImageMaxBytes: OPENAI_CODEX_REQUEST_IMAGE_MAX_BYTES,
    piProvider: responses.wrap(requestProvider(provider, fastMode)),
  }]])
  const models: MutableModels = createModels({ credentials })
  models.setProvider(provider)
  return new OpenAICodexAdapter({
    profiles: () => profiles,
    resolveApiKey: async () => (await models.getAuth(OPENAI_CODEX_PROVIDER))?.auth.apiKey,
    auth: { credentials, authContext: PROCESS_AUTH_CONTEXT },
    resolveAttachments,
  }, responses, visibleModelIds)
}
