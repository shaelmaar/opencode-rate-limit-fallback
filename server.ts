/**
 * OpenCode v2 plugin entry for local directory loading.
 *
 * OpenCode's loader for a configured local plugin directory does not follow
 * package.json "main"/"exports": it probes `server` and then `index` file
 * names at the directory root. The root `index.ts` is the legacy OpenCode v1
 * plugin, so this file gives the v2 host the v2 definition first.
 */
export { default } from "./v2.js";
