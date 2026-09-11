import { describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import { SettingsProvider } from '@deepseek-ai/dsh-settings'
import { Config } from '../src/index.ts'
import { OpenAICodexService } from '../src/service.ts'
import { openAICodexAuthPath } from '../src/store.ts'
import { resolve } from 'node:path'

const preferences = {
  modifyReadImage: true,
  shareImagegenWithOtherModels: true,
  useWebSocketContextReuse: false,
  useNativeCompaction: false,
  contextWindow: null,
  overrideSparkContextWindow: false,
  fastModeDefault: false,
  proxyMode: 'off' as const,
  proxyUrl: '',
  modelCatalog: [],
}

describe('credentialFile configuration seam', () => {
  it('passes the explicit credential filename into the shared service store', () => {
    const filename = resolve('fixture-auth.json')
    const config = new Config({ credentialFile: filename })
    const service = new OpenAICodexService({ ...preferences, credentialFile: filename })
    expect(config.credentialFile).toBe(filename)
    expect(service.credentials.filename).toBe(filename)
  })

  it('retains the DSH_HOME-derived filename when credentialFile is absent', () => {
    const config = new Config()
    const service = new OpenAICodexService(preferences)
    expect(config.credentialFile).toBeUndefined()
    expect(service.credentials.filename).toBe(openAICodexAuthPath())
  })
})

class MemorySettings extends SettingsProvider {
  readonly writable = false

  protected load(): Promise<Record<string, unknown>> {
    return Promise.resolve({})
  }

  protected persist(): Promise<void> {
    return Promise.reject(new Error('Settings are read-only'))
  }
}

describe('service settings boundary', () => {
  it.each(['static', 'dynamic'] as const)(
    'keeps runtime options out of settings with a %s model catalog',
    async (mode) => {
      const ctx = new Context()
      let catalog = [
        { id: 'gpt-current', name: 'GPT Current', contextWindow: 272_000 },
      ]
      const models = ['gpt-current', 'gpt-future']
      const service = new OpenAICodexService({
        ...preferences,
        credentialFile: resolve('fixture-auth.json'),
        modifyReadImage: false,
        fastModeDefault: true,
        models,
        modelCatalog: mode === 'dynamic' ? () => catalog : catalog,
      })
      try {
        await ctx.plugin(MemorySettings)
        service.attachSettings(ctx)

        for (const redactSecrets of [false, true]) {
          const descriptors = ctx.settings.describe({ redactSecrets })
          expect(descriptors).toHaveLength(1)
          const descriptor = descriptors[0]!
          expect(descriptor).toMatchObject({
            ns: 'openai-codex',
            base: { modifyReadImage: false, fastModeDefault: true, models },
            value: { modifyReadImage: false, fastModeDefault: true, models },
          })
          expect(descriptor.base).not.toHaveProperty('modelCatalog')
          expect(descriptor.base).not.toHaveProperty('credentialFile')
          expect(structuredClone(descriptor)).toEqual(descriptor)
        }
        expect(service.modelCatalogSettings()).toEqual({
          availableModels: catalog,
          models: ['gpt-current'],
        })

        if (mode === 'dynamic') {
          catalog = [
            ...catalog,
            { id: 'gpt-future', name: 'GPT Future', contextWindow: 512_000 },
          ]
          expect(service.modelCatalogSettings()).toEqual({
            availableModels: catalog,
            models,
          })
        }
      } finally {
        await service.dispose()
        await ctx.fiber.dispose()
      }
    },
  )
})
