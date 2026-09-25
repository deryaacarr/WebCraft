const INCLUDE_RE = /^[ \t]*#include[ \t]+"([^"]+)"[ \t]*$/gm;

/**
 * Expands `#include "file.wgsl"` directives. Each file is included at most once
 * per shader (include-guard semantics); cycles and missing files throw.
 */
export function preprocessWgsl(
  entry: string,
  resolve: (name: string) => string | undefined,
): string {
  const included = new Set<string>();

  const expand = (name: string, stack: string[]): string => {
    if (stack.includes(name)) {
      throw new Error(`WGSL include cycle: ${[...stack, name].join(' -> ')}`);
    }
    if (included.has(name)) return '';
    const source = resolve(name);
    if (source === undefined) {
      const from = stack.at(-1);
      throw new Error(`WGSL include not found: "${name}"${from ? ` (from ${from})` : ''}`);
    }
    included.add(name);
    const next = [...stack, name];
    return source.replace(INCLUDE_RE, (_line, dep: string) => expand(dep, next));
  };

  return expand(entry, []);
}
