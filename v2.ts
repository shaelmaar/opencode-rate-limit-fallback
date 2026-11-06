import { loadConfig } from "./src/utils/config.js";
import type { FallbackModel } from "./src/types/index.js";

type RetryError = {
  readonly type?: string;
  readonly message?: string;
  readonly status?: number;
};

type ModelRef = {
  readonly providerID: string;
  readonly id: string;
  readonly variant?: string;
};

type RetryEvent = {
  readonly sessionID: string;
  /** Model of the request that failed. Present on OpenCode v2 GA. */
  readonly model?: ModelRef;
  readonly error: RetryError;
  readonly attempt: number;
  decision: { retry: false } | { retry: true; delay: number };
};

type PromptEvent = {
  readonly sessionID: string;
  readonly prompt: { readonly text: string };
};

type Registration = { readonly dispose: () => Promise<void> | void };

type V2Context = {
  readonly location: { readonly directory: string };
  readonly session: {
    hook(
      name: "prompt",
      callback: (event: PromptEvent) => Promise<void> | void,
    ): Promise<Registration>;
    hook(
      name: "retry",
      callback: (event: RetryEvent) => Promise<void> | void,
    ): Promise<Registration>;
    prompt(input: {
      sessionID: string;
      text: string;
      resume: boolean;
    }): Promise<unknown>;
    // Optional so a host without model switching still gets wait behaviour.
    switchModel?(input: { sessionID: string; model: ModelRef }): Promise<unknown>;
    get?(input: { sessionID: string }): Promise<{ readonly model?: ModelRef } | undefined>;
  };
  readonly event?: {
    subscribe(options?: { signal?: AbortSignal }): AsyncIterable<SessionEvent>;
  };
};

type SessionEvent = {
  readonly type: string;
  readonly data?: { readonly sessionID?: string; readonly error?: RetryError };
};

const MAX_BACKOFF_MS = 60 * 60 * 1000;

/** Delay before retrying on a freshly selected fallback model. */
const SWITCH_DELAY_MS = 1000;

const RETRYABLE_ERROR_PATTERNS = [
  /\b429\b/i,
  /rate[\s_-]*limit/i,
  /too many requests/i,
  /quota/i,
  /usage limit/i,
  /resource[\s_-]*exhausted/i,
  /try again later/i,
  // Transient upstream failures that OpenCode classifies as ProviderInternal
  // (for example "Error from provider (Console Go): Upstream request failed:
  // Endpoint is unavailable."). These clear on their own, so wait them out
  // instead of letting the turn stop once OpenCode's own retry budget runs out.
  /endpoint is unavailable/i,
  /upstream request failed/i,
];

function isRetryableError(value: unknown): boolean {
  if (!value || typeof value !== "object") return false;
  const error = value as RetryError;
  if (error.status === 429) return true;
  const text = `${error.type ?? ""} ${error.message ?? ""}`;
  return RETRYABLE_ERROR_PATTERNS.some((pattern) => pattern.test(text));
}

/** Keys whose string values are human-readable error text worth matching. */
const ERROR_TEXT_KEYS = ["message", "type", "code", "responseBody", "statusText", "detail", "details", "reason", "title"] as const;

/** Keys whose values may nest further error information. */
const ERROR_NEST_KEYS = ["cause", "data", "response", "body", "error", "errors"] as const;

// Caps that bound the cost of matching user-supplied regular expressions.
const MAX_ADDITIONAL_PATTERNS = 50;
const MAX_PATTERN_SOURCE_LENGTH = 256;
const MAX_ERROR_TEXT_LENGTH = 8192;
const MAX_ERROR_TEXT_DEPTH = 6;
const MAX_ERROR_ARRAY_ITEMS = 20;

/**
 * Compile `errorPatterns.additionalPatterns` into case-insensitive regexes.
 * Entries that are not non-empty strings, or do not compile, are skipped so
 * one broken rule never disables the rest or crashes the plugin.
 */
function compileAdditionalPatterns(patterns: readonly unknown[] | undefined): RegExp[] {
  if (!Array.isArray(patterns)) return [];
  const compiled: RegExp[] = [];
  const seen = new Set<string>();
  for (const entry of patterns) {
    if (compiled.length >= MAX_ADDITIONAL_PATTERNS) break;
    if (typeof entry !== "string") continue;
    const source = entry.trim();
    if (source.length === 0 || source.length > MAX_PATTERN_SOURCE_LENGTH) continue;
    const key = source.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    try {
      compiled.push(new RegExp(source, "i"));
    } catch {
      // Not a valid regular expression: skip it, keep the other rules.
    }
  }
  return compiled;
}

function collectErrorText(value: unknown, depth: number, seen: Set<object>, parts: string[]): void {
  if (typeof value === "string") {
    if (value.length > 0) parts.push(value);
    return;
  }
  // Never serialize the whole object: only known human-readable keys are read,
  // so unrelated fields (and any secrets they hold) cannot be matched or logged.
  if (value === null || typeof value !== "object" || depth <= 0 || seen.has(value)) return;
  seen.add(value);
  if (Array.isArray(value)) {
    for (const item of value.slice(0, MAX_ERROR_ARRAY_ITEMS)) {
      collectErrorText(item, depth - 1, seen, parts);
    }
    return;
  }
  const record = value as Record<string, unknown>;
  for (const key of ERROR_TEXT_KEYS) {
    const text = record[key];
    if (typeof text === "string" && text.length > 0) parts.push(text);
  }
  for (const key of ERROR_NEST_KEYS) {
    if (key in record) collectErrorText(record[key], depth - 1, seen, parts);
  }
}

/**
 * The human-readable text of an error: its message-ish fields plus the same
 * fields of objects nested under `cause`, `data`, `response`, ... Cyclic
 * references and nesting depth are bounded.
 */
function errorText(value: unknown): string {
  const parts: string[] = [];
  collectErrorText(value, MAX_ERROR_TEXT_DEPTH, new Set(), parts);
  return parts.join(" ").slice(0, MAX_ERROR_TEXT_LENGTH);
}

/**
 * True when the error text matches a user-configured pattern: the model is
 * unavailable on it (for example a subscription restriction), so fail over
 * without retrying it.
 */
function isFallbackOnlyError(value: unknown, patterns: readonly RegExp[]): boolean {
  if (patterns.length === 0) return false;
  const text = errorText(value);
  return text.length > 0 && patterns.some((pattern) => pattern.test(text));
}

function effectiveCooldown(cooldownMs: number | undefined): number {
  return Math.max(1000, cooldownMs ?? 60000);
}

function exponentialDelay(baseDelayMs: number, failureCount: number): number {
  const exponent = Math.min(Math.max(0, failureCount - 1), 52);
  return Math.min(Math.max(MAX_BACKOFF_MS, baseDelayMs), baseDelayMs * 2 ** exponent);
}

function sameModel(a: ModelRef | undefined, b: ModelRef | undefined): boolean {
  // Variant (reasoning-effort selector) is deliberately ignored when matching:
  // the chain is keyed by provider + model, so a session on glm-5.3#high must
  // still match a variant-less glm-5.3 chain entry and fall back.
  return a !== undefined && b !== undefined && a.providerID === b.providerID && a.id === b.id;
}

function toModelRef(model: FallbackModel): ModelRef {
  return { providerID: model.providerID, id: model.modelID };
}

/**
 * The models a session walks through on consecutive failures: its own model,
 * then the configured fallbacks listed after it. A model that is not in
 * `fallbackModels` has no fallbacks: it is retried after a cooldown when
 * otherModels is "wait" (or in "wait" mode), and otherwise never gets here.
 */
function chainFor(model: ModelRef | undefined, fallbacks: readonly ModelRef[]): ModelRef[] {
  if (model === undefined) return [];
  const index = fallbacks.findIndex((candidate) => sameModel(candidate, model));
  if (index < 0) return [model];
  // Carry the failed model's reasoning-effort variant onto fallbacks of the
  // SAME model so "same model, same effort" holds across providers (e.g.
  // zai-coding-plan/glm-5.3#high -> zai/glm-5.3#high). Different-model tiers
  // (e.g. a flash model) keep their own default effort.
  const variant = model.variant;
  const carry = (candidate: ModelRef): ModelRef =>
    variant && !candidate.variant && candidate.id === model.id ? { ...candidate, variant } : candidate;
  return [model, ...fallbacks.slice(index + 1).filter((candidate) => !sameModel(candidate, model)).map(carry)];
}

/** One run of consecutive failures in a session, ended by a real user prompt. */
type Streak = {
  /** The session's model when the streak began; restored on the next real prompt. */
  original: ModelRef | undefined;
  chain: ModelRef[];
  /** Full passes through the chain that ended in failure; drives the backoff. */
  passes: number;
  /** The model this plugin last selected, if it switched at all. */
  selected?: ModelRef;
};

export default {
  id: "opencode-rate-limit-fallback",
  async setup(context: V2Context) {
    const { config } = loadConfig(context.location.directory, context.location.directory);
    if (!config.enabled) return;

    // "wait" never switches models; the other modes walk the fallback chain
    // first and only then decide between waiting ("cycle", "retry-last") and
    // giving up ("stop"). A session whose model is not in the chain is left
    // to OpenCode unless otherModels is "wait".
    const mode = config.fallbackMode;
    const otherModels = config.otherModels ?? "ignore";
    const fallbacks = mode === "wait" ? [] : (config.fallbackModels ?? [])
      .filter((model) => typeof model?.providerID === "string" && typeof model?.modelID === "string")
      .map(toModelRef);
    if (mode !== "wait" && fallbacks.length === 0 && otherModels !== "wait") return;
    const handles = (model: ModelRef | undefined) =>
      mode === "wait" || otherModels === "wait" || fallbacks.some((candidate) => sameModel(candidate, model));
    const cooldownMs = effectiveCooldown(config.cooldownMs);
    // Compiled once here: like the rest of the config, the patterns are read
    // at setup and a change takes effect the next time the plugin is loaded.
    const additionalPatterns = compileAdditionalPatterns(config.errorPatterns?.additionalPatterns);
    const resumeTimers = new Map<string, ReturnType<typeof setTimeout>>();
    const streaks = new Map<string, Streak>();

    const cancelResume = (sessionID: string) => {
      const previous = resumeTimers.get(sessionID);
      if (previous) {
        clearTimeout(previous);
        resumeTimers.delete(sessionID);
      }
    };

    const scheduleResume = (sessionID: string, delay: number) => {
      cancelResume(sessionID);
      const timer = setTimeout(() => {
        resumeTimers.delete(sessionID);
        void context.session.prompt({ sessionID, text: "", resume: true }).catch(() => {
          // A deleted or otherwise unavailable session must not keep retrying.
        });
      }, delay);
      resumeTimers.set(sessionID, timer);
    };

    const switchModel = async (sessionID: string, model: ModelRef): Promise<boolean> => {
      if (!context.session.switchModel) return false;
      try {
        await context.session.switchModel({ sessionID, model });
        return true;
      } catch {
        return false;
      }
    };

    const currentModel = async (sessionID: string): Promise<ModelRef | undefined> => {
      try {
        return context.session.get ? (await context.session.get({ sessionID }))?.model : undefined;
      } catch {
        return undefined;
      }
    };

    // Put the session back on its own model, unless someone picked another
    // model since this plugin switched it.
    const restoreModel = async (sessionID: string, streak: Streak) => {
      if (streak.selected === undefined || streak.original === undefined) return;
      if (sameModel(streak.selected, streak.original)) return;
      const current = await currentModel(sessionID);
      if (current !== undefined && !sameModel(current, streak.selected)) return;
      await switchModel(sessionID, streak.original);
    };

    // `failed` just failed with a retryable error: select the model for the
    // next try and say when it is due. `fallbackOnly` marks an error whose
    // text matched a user pattern: the model is unavailable, so when there is
    // no next model the turn ends instead of waiting or cycling back.
    const decide = async (
      sessionID: string,
      failed: ModelRef | undefined,
      fallbackOnly = false,
    ): Promise<RetryEvent["decision"]> => {
      let streak = streaks.get(sessionID);
      let position = streak?.chain.findIndex((model) => sameModel(model, failed)) ?? -1;
      if (!streak || (failed !== undefined && position < 0)) {
        // First failure, or the model was changed by someone else mid-streak:
        // start over from the model that just failed.
        streak = { original: failed, chain: chainFor(failed, fallbacks), passes: streak?.passes ?? 0 };
        streaks.set(sessionID, streak);
        position = 0;
      }

      let next: ModelRef | undefined = streak.chain[position + 1];
      let delay = SWITCH_DELAY_MS;
      if (next !== undefined && !(await switchModel(sessionID, next))) next = undefined;
      if (next === undefined) {
        // Every model in the chain failed on this pass (or switching is not possible).
        if (fallbackOnly) {
          // The error text says the model is unavailable (for example a
          // subscription restriction): retrying or cycling back to it cannot
          // help, so end the turn, even in "cycle" mode.
          return { retry: false };
        }
        if (mode === "stop" && streak.chain.length > 1) return { retry: false };
        streak.passes += 1;
        delay = exponentialDelay(cooldownMs, streak.passes);
        const first = streak.chain[0];
        if (mode !== "retry-last" && first !== undefined && !sameModel(first, failed)) {
          if (await switchModel(sessionID, first)) next = first;
        }
      }
      if (next !== undefined) streak.selected = next;
      return { retry: true, delay };
    };

    const promptRegistration = await context.session.hook("prompt", async (event) => {
      // A real user prompt starts a fresh backoff sequence on the session's own
      // model. The empty prompt used to resume a failed turn must preserve the
      // streak.
      if (event.prompt.text.trim().length === 0) return;
      cancelResume(event.sessionID);
      const streak = streaks.get(event.sessionID);
      streaks.delete(event.sessionID);
      if (streak) await restoreModel(event.sessionID, streak);
    });

    const retryRegistration = await context.session.hook("retry", async (event) => {
      // A user-configured pattern wins over the retry classification: the
      // error text says this model is unavailable, whatever the status code.
      const fallbackOnly = isFallbackOnlyError(event.error, additionalPatterns);
      if ((!fallbackOnly && !isRetryableError(event.error)) || !handles(event.model)) return;
      cancelResume(event.sessionID);
      event.decision = await decide(event.sessionID, event.model, fallbackOnly);
    });

    // OpenCode has its own ceiling on retries and fails the turn once it is
    // reached, even when this plugin asked for another try. Such a turn is
    // resumed when the next try is due. An interrupted turn stays stopped.
    const abort = new AbortController();
    const onEvent = async (event: SessionEvent) => {
      const sessionID = event.data?.sessionID;
      if (typeof sessionID !== "string") return;
      if (event.type === "session.deleted") {
        cancelResume(sessionID);
        streaks.delete(sessionID);
        return;
      }
      if (event.type !== "session.execution.failed") return;
      const streak = streaks.get(sessionID);
      if (!streak) return;
      const fallbackOnly = isFallbackOnlyError(event.data?.error, additionalPatterns);
      if (!fallbackOnly && !isRetryableError(event.data?.error)) return;
      const failed = (await currentModel(sessionID)) ?? streak.selected ?? streak.original;
      if (!handles(failed)) return;
      const decision = await decide(sessionID, failed, fallbackOnly);
      if (decision.retry) scheduleResume(sessionID, decision.delay);
    };
    const watch = async (events: NonNullable<V2Context["event"]>) => {
      while (!abort.signal.aborted) {
        try {
          for await (const event of events.subscribe({ signal: abort.signal })) {
            void onEvent(event).catch(() => {});
          }
        } catch {
          // The stream broke; subscribe again below.
        }
        if (!abort.signal.aborted) await new Promise((resolve) => setTimeout(resolve, 1000));
      }
    };
    if (context.event) void watch(context.event);

    return async () => {
      abort.abort();
      await promptRegistration.dispose();
      await retryRegistration.dispose();
      for (const timer of resumeTimers.values()) clearTimeout(timer);
      resumeTimers.clear();
      streaks.clear();
    };
  },
};
