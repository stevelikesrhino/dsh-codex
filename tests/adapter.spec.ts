import { describe, expect, it } from 'vitest'
import { AttachmentId } from '@deepseek-ai/dsh-attachment'
import type { AttachmentStore, ImageAttachmentRef, ImageRequestPolicy } from '@deepseek-ai/dsh-attachment'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import type { OpenAICodexCredentialStore } from '../src/store.ts'
import { OPENAI_CODEX_PROVIDER } from '../src/store.ts'
import {
  createOpenAICodexAdapter,
  OPENAI_CODEX_MAX_REQUEST_IMAGE_BYTES,
  OPENAI_CODEX_REQUEST_IMAGE_MAX_BYTES,
  OPENAI_CODEX_REQUEST_IMAGE_PIXEL_BUDGET,
  OPENAI_CODEX_RETRY_POLICY,
} from '../src/adapter.ts'
import { Config } from '../src/index.ts'

const IMAGE_REF: ImageAttachmentRef = {
  attachmentId: AttachmentId(`sha256:${'a'.repeat(64)}`),
  mediaType: 'image/png',
  bytes: 1,
  width: 1,
  height: 1,
}

describe('OpenAI Codex adapter policy', () => {
  it('distinguishes an omitted model list from an explicitly empty list', () => {
    expect(Config({}).models).toBeUndefined()
    expect(Config({ models: [] }).models).toEqual([])
  })

  it('registers the extended bounded retry policy on the provider route', () => {
    const adapter = createOpenAICodexAdapter(
      {} as OpenAICodexCredentialStore,
      () => undefined,
      () => ({ useWebSocketContextReuse: false, useNativeCompaction: false }),
    )

    expect(adapter.providerRetryPolicy(OPENAI_CODEX_PROVIDER)).toBe(OPENAI_CODEX_RETRY_POLICY)
    expect(OPENAI_CODEX_RETRY_POLICY).toMatchObject({
      mode: 'normal',
      maxRetries: 5,
      retryableCodes: expect.arrayContaining(['RATE_LIMIT', 'SERVER', 'TIMEOUT', 'TRANSPORT']),
      initialDelayMs: 1_000,
      maxDelayMs: 30_000,
      jitterRatio: 0.2,
    })
  })

  it('advertises only configured models while keeping hidden models resolvable', async () => {
    const adapter = createOpenAICodexAdapter(
      {} as OpenAICodexCredentialStore,
      () => undefined,
      () => ({ useWebSocketContextReuse: false, useNativeCompaction: false }),
      undefined,
      () => ['gpt-5.6-terra', 'gpt-5.6-luna', 'gpt-5.6-terra'],
    )

    const models = await adapter.listModels(OPENAI_CODEX_PROVIDER)
    expect(models.map(model => model.id)).toEqual(['gpt-5.6-luna', 'gpt-5.6-terra'])

    await expect(adapter.resolveModel(OPENAI_CODEX_PROVIDER, 'gpt-5.4')).resolves.toMatchObject({
      provider: OPENAI_CODEX_PROVIDER,
      id: 'gpt-5.4',
    })
  })

  it('advertises the full provider catalog when no model list is configured', async () => {
    const adapter = createOpenAICodexAdapter(
      {} as OpenAICodexCredentialStore,
      () => undefined,
      () => ({ useWebSocketContextReuse: false, useNativeCompaction: false }),
    )

    const models = await adapter.listModels(OPENAI_CODEX_PROVIDER)
    expect(models.map(model => model.id)).toEqual(expect.arrayContaining([
      'gpt-5.4',
      'gpt-5.6-luna',
      'gpt-5.6-sol',
      'gpt-5.6-terra',
    ]))
  })

  it('resolves request images with the dsh default route budgets', async () => {
    const policies: ImageRequestPolicy[] = []
    const sentinel = new Error('request image policy observed')
    const attachments = {
      readImageRequest: async (_ref: ImageAttachmentRef, policy: ImageRequestPolicy): Promise<never> => {
        policies.push(policy)
        throw sentinel
      },
    } as unknown as AttachmentStore
    const adapter = createOpenAICodexAdapter(
      { read: async () => undefined } as unknown as OpenAICodexCredentialStore,
      () => attachments,
      () => ({ useWebSocketContextReuse: false, useNativeCompaction: false }),
    )

    const drain = async (): Promise<void> => {
      for await (const _chunk of adapter.stream({
        provider: OPENAI_CODEX_PROVIDER,
        model: 'gpt-5.6-luna',
        messages: [createUserMessage({
          content: [{ type: 'image', attachment: IMAGE_REF }],
          source: { kind: 'plugin', plugin: 'test' },
        })],
      })) { /* drain */ }
    }

    await expect(drain()).rejects.toBe(sentinel)
    expect(policies).toEqual([{
      maxPixels: OPENAI_CODEX_REQUEST_IMAGE_PIXEL_BUDGET,
      maxBytes: OPENAI_CODEX_REQUEST_IMAGE_MAX_BYTES,
    }])
    expect(OPENAI_CODEX_MAX_REQUEST_IMAGE_BYTES).toBe(20 * 1024 * 1024)
    expect(OPENAI_CODEX_REQUEST_IMAGE_PIXEL_BUDGET).toBe(2048 * 2048)
    expect(OPENAI_CODEX_REQUEST_IMAGE_MAX_BYTES).toBe(1024 * 1024)
  })
})
