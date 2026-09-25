import { describe, expect, it } from 'vitest';
import { preprocessWgsl } from './preprocess';

const files: Record<string, string> = {
  'main.wgsl': '#include "a.wgsl"\n#include "b.wgsl"\nfn main() {}',
  'a.wgsl': '#include "common.wgsl"\nfn a() {}',
  'b.wgsl': '  #include "common.wgsl"\nfn b() {}',
  'common.wgsl': 'const PI = 3.14;',
  'cycle1.wgsl': '#include "cycle2.wgsl"',
  'cycle2.wgsl': '#include "cycle1.wgsl"',
  'missing.wgsl': '#include "nope.wgsl"',
  'comment.wgsl': '// #include "nope.wgsl"\nfn c() {}',
};
const resolve = (n: string) => files[n];

describe('preprocessWgsl', () => {
  it('expands nested includes and includes each file once', () => {
    const out = preprocessWgsl('main.wgsl', resolve);
    expect(out.match(/const PI/g)).toHaveLength(1);
    expect(out.indexOf('const PI')).toBeLessThan(out.indexOf('fn a()'));
    expect(out).toContain('fn b()');
    expect(out).not.toContain('#include');
  });

  it('throws on include cycles', () => {
    expect(() => preprocessWgsl('cycle1.wgsl', resolve)).toThrow(/cycle/);
  });

  it('throws on missing files with the including file in the message', () => {
    expect(() => preprocessWgsl('missing.wgsl', resolve)).toThrow(/nope\.wgsl.*missing\.wgsl/);
  });

  it('ignores directives that are not at line start', () => {
    expect(preprocessWgsl('comment.wgsl', resolve)).toContain('// #include "nope.wgsl"');
  });
});
