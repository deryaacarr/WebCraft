/**
 * Which textures every material uses. Single source for fetch-textures.ts (downloads,
 * CREDITS.md) and build-textures.ts (packing). All downloaded assets are ambientCG
 * (CC0 1.0 Universal, https://docs.ambientcg.com/license/).
 */
import type { MaterialName } from '../src/world/blocks.ts';

/** An ambientCG material. `crop`: fraction of the source's edge that covers one block
 *  face (1 m). Stochastic textures are cropped to real-world scale and re-tiled by edge
 *  blending; structured ones (paving, planks) keep crop 1 and their native tiling. */
export interface AmbientCgSource {
  kind: 'ambientcg';
  id: string;
  crop: number;
}

/** Dirt with a band of grass along the top edge (side face of a grass block). */
export interface GrassSideSource {
  kind: 'grass-side';
  dirt: string;
  grass: string;
  seed: number;
}

/** Leaves scattered from single-leaf ambientCG sprites onto a transparent, tileable canvas. */
export interface LeavesSource {
  kind: 'leaves';
  sprites: string[];
  seed: number;
}

export interface ProceduralSource {
  kind: 'procedural';
  generator: 'water' | 'glass' | 'torch' | 'log-top';
  seed: number;
}

export type VariantSource = AmbientCgSource | GrassSideSource | LeavesSource | ProceduralSource;

export interface MaterialSpec {
  variants: VariantSource[];
  /** Random 90° rotation per voxel face (off for textures with a direction, e.g. bark). */
  rotate: boolean;
  /** Parallax occlusion mapping from the height channel. */
  pom: boolean;
  /** Pixels with opacity below the cutoff let rays through (leaves). */
  alphaTest: boolean;
  /** Natural material: also packed as world-space layers covering WORLD_SPAN × WORLD_SPAN
   *  metres (sampled from world coordinates, no per-block grid). */
  world?: boolean;
}

/** Edge (m) of the area one world-space layer covers (same texel density as block layers). */
export const WORLD_SPAN = 2;

const acg = (id: string, crop: number): AmbientCgSource => ({ kind: 'ambientcg', id, crop });

export const TEXTURE_SPEC: Record<MaterialName, MaterialSpec> = {
  grass_top: { variants: [acg('Grass001', 0.7), acg('Grass004', 0.7), acg('Grass005', 0.6)], rotate: true, pom: false, alphaTest: false },
  grass_side: {
    variants: [
      { kind: 'grass-side', dirt: 'Ground048', grass: 'Grass001', seed: 1 },
      { kind: 'grass-side', dirt: 'Ground023', grass: 'Grass004', seed: 2 },
      { kind: 'grass-side', dirt: 'Ground048', grass: 'Grass005', seed: 3 },
    ],
    rotate: false,
    pom: false,
    alphaTest: false,
  },
  dirt: { variants: [acg('Ground048', 0.7), acg('Ground023', 0.5)], rotate: true, pom: true, alphaTest: false, world: true },
  stone: { variants: [acg('Rock030', 0.5), acg('Rock050', 0.5), acg('Rock060', 0.5)], rotate: true, pom: false, alphaTest: false, world: true },
  cobblestone: { variants: [acg('PavingStones119', 1), acg('PavingStones046', 1)], rotate: true, pom: true, alphaTest: false },
  gravel: { variants: [acg('Gravel023', 0.67), acg('Gravel040', 0.5)], rotate: true, pom: true, alphaTest: false, world: true },
  sand: { variants: [acg('Ground054', 0.3), acg('Ground080', 0.5)], rotate: true, pom: false, alphaTest: false, world: true },
  water: { variants: [{ kind: 'procedural', generator: 'water', seed: 1 }], rotate: true, pom: false, alphaTest: false },
  oak_log_top: {
    variants: [
      { kind: 'procedural', generator: 'log-top', seed: 1 },
      { kind: 'procedural', generator: 'log-top', seed: 2 },
    ],
    rotate: true,
    pom: false,
    alphaTest: false,
  },
  oak_log_side: { variants: [acg('Bark012', 0.5), acg('Bark007', 0.5)], rotate: false, pom: false, alphaTest: false },
  oak_leaves: {
    variants: [
      { kind: 'leaves', sprites: ['Leaf001', 'Leaf003'], seed: 1 },
      { kind: 'leaves', sprites: ['Leaf001', 'Leaf003'], seed: 2 },
      { kind: 'leaves', sprites: ['Leaf003', 'Leaf001'], seed: 3 },
    ],
    rotate: true,
    pom: false,
    alphaTest: true,
  },
  oak_planks: { variants: [acg('Planks037A', 1)], rotate: false, pom: false, alphaTest: false },
  glass: { variants: [{ kind: 'procedural', generator: 'glass', seed: 1 }], rotate: false, pom: false, alphaTest: false },
  torch: { variants: [{ kind: 'procedural', generator: 'torch', seed: 1 }], rotate: false, pom: false, alphaTest: false },
  lava: { variants: [acg('Lava001', 0.5), acg('Lava004', 0.5)], rotate: true, pom: false, alphaTest: false },
};

/** Texture resolutions the packer produces (px per block face). 256 = "Ultra". */
export const TEXTURE_RESOLUTIONS = [16, 32, 64, 128, 256] as const;

/** Every ambientCG asset id the spec references. */
export function ambientCgIds(): string[] {
  const ids = new Set<string>();
  for (const spec of Object.values(TEXTURE_SPEC)) {
    for (const v of spec.variants) {
      if (v.kind === 'ambientcg') ids.add(v.id);
      else if (v.kind === 'grass-side') {
        ids.add(v.dirt);
        ids.add(v.grass);
      } else if (v.kind === 'leaves') v.sprites.forEach((s) => ids.add(s));
    }
  }
  return [...ids].sort();
}
