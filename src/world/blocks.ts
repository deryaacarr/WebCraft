/**
 * Block registry. Block ids are uint16 values stored directly in chunk data,
 * so the order of `BlockId` is part of the save/GPU format: append, never reorder.
 */

/** Surface materials (future texture array layers). Order = material index. */
export const MATERIAL_NAMES = [
  'grass_top',
  'grass_side',
  'dirt',
  'stone',
  'cobblestone',
  'gravel',
  'sand',
  'water',
  'oak_log_top',
  'oak_log_side',
  'oak_leaves',
  'oak_planks',
  'glass',
  'torch',
  'lava',
] as const;

export type MaterialName = (typeof MATERIAL_NAMES)[number];

/** Approximate sRGB albedo per material (0xRRGGBB) until textures exist. */
export const MATERIAL_COLORS: Record<MaterialName, number> = {
  grass_top: 0x5f9e3a,
  grass_side: 0x6e7a3c,
  dirt: 0x7a5534,
  stone: 0x7f7f7f,
  cobblestone: 0x6b6b6b,
  gravel: 0x8a827a,
  sand: 0xd8c894,
  water: 0x2f5fb0,
  oak_log_top: 0x9c7a4a,
  oak_log_side: 0x6b4f2a,
  oak_leaves: 0x3c7a2a,
  oak_planks: 0xa9824f,
  glass: 0xcfe8f0,
  torch: 0xffc050,
  lava: 0xff5a10,
};

/** Material index used for faces that are never drawn (air). */
export const NO_MATERIAL = 0xffff;

export function materialIndex(name: MaterialName): number {
  return MATERIAL_NAMES.indexOf(name);
}

export type FootstepMaterial = 'grass' | 'stone' | 'wood' | 'sand' | 'gravel';

export const BlockId = {
  air: 0,
  grass: 1,
  dirt: 2,
  stone: 3,
  cobblestone: 4,
  gravel: 5,
  sand: 6,
  water: 7,
  oak_log: 8,
  oak_leaves: 9,
  oak_planks: 10,
  glass: 11,
  torch: 12,
  lava: 13,
} as const;

export type BlockName = keyof typeof BlockId;
export type BlockId = (typeof BlockId)[BlockName];

export interface FaceMaterials {
  top: number;
  side: number;
  bottom: number;
}

export interface BlockDef {
  id: BlockId;
  name: BlockName;
  /** Collides with the player and occludes neighbours. */
  solid: boolean;
  /** Light passes through (fully or partially); neighbours' faces stay visible. */
  transparent: boolean;
  /** Emitted radiance, relative units (0 = none). Calibrated once lighting/tonemapping exist. */
  emissive: number;
  materials: FaceMaterials;
  /** Footstep sound set; null for blocks you cannot stand on. */
  footstep: FootstepMaterial | null;
  /** Approximate sRGB albedo (0xRRGGBB), for debug views until textures exist. */
  color: number;
}

interface BlockSpec {
  solid: boolean;
  transparent: boolean;
  emissive: number;
  /** One material for all faces, or per-face materials. */
  faces: MaterialName | { top: MaterialName; side: MaterialName; bottom: MaterialName } | null;
  footstep: FootstepMaterial | null;
  color: number;
}

const SPECS = {
  air: { solid: false, transparent: true, emissive: 0, faces: null, footstep: null, color: 0x000000 },
  grass: {
    solid: true,
    transparent: false,
    emissive: 0,
    faces: { top: 'grass_top', side: 'grass_side', bottom: 'dirt' },
    footstep: 'grass', color: 0x5f9e3a,
  },
  dirt: { solid: true, transparent: false, emissive: 0, faces: 'dirt', footstep: 'grass', color: 0x7a5534 },
  stone: { solid: true, transparent: false, emissive: 0, faces: 'stone', footstep: 'stone', color: 0x7f7f7f },
  cobblestone: { solid: true, transparent: false, emissive: 0, faces: 'cobblestone', footstep: 'stone', color: 0x6b6b6b },
  gravel: { solid: true, transparent: false, emissive: 0, faces: 'gravel', footstep: 'gravel', color: 0x8a827a },
  sand: { solid: true, transparent: false, emissive: 0, faces: 'sand', footstep: 'sand', color: 0xd8c894 },
  water: { solid: false, transparent: true, emissive: 0, faces: 'water', footstep: null, color: 0x2f5fb0 },
  oak_log: {
    solid: true,
    transparent: false,
    emissive: 0,
    faces: { top: 'oak_log_top', side: 'oak_log_side', bottom: 'oak_log_top' },
    footstep: 'wood', color: 0x6b4f2a,
  },
  oak_leaves: { solid: true, transparent: true, emissive: 0, faces: 'oak_leaves', footstep: 'grass', color: 0x3c7a2a },
  oak_planks: { solid: true, transparent: false, emissive: 0, faces: 'oak_planks', footstep: 'wood', color: 0xa9824f },
  glass: { solid: true, transparent: true, emissive: 0, faces: 'glass', footstep: 'stone', color: 0xcfe8f0 },
  torch: { solid: false, transparent: true, emissive: 8, faces: 'torch', footstep: null, color: 0xffc050 },
  lava: { solid: false, transparent: false, emissive: 4, faces: 'lava', footstep: null, color: 0xff5a10 },
} satisfies Record<BlockName, BlockSpec>;

function toFaceMaterials(faces: BlockSpec['faces']): FaceMaterials {
  if (faces === null) return { top: NO_MATERIAL, side: NO_MATERIAL, bottom: NO_MATERIAL };
  if (typeof faces === 'string') {
    const m = materialIndex(faces);
    return { top: m, side: m, bottom: m };
  }
  return {
    top: materialIndex(faces.top),
    side: materialIndex(faces.side),
    bottom: materialIndex(faces.bottom),
  };
}

/** Registry indexed by block id. */
export const BLOCKS: readonly BlockDef[] = (Object.keys(BlockId) as BlockName[])
  .map((name): BlockDef => {
    const spec: BlockSpec = SPECS[name];
    return {
      id: BlockId[name],
      name,
      solid: spec.solid,
      transparent: spec.transparent,
      emissive: spec.emissive,
      materials: toFaceMaterials(spec.faces),
      footstep: spec.footstep,
      color: spec.color,
    };
  })
  .sort((a, b) => a.id - b.id);

// Flat lookup tables for hot loops (physics, packing).
const SOLID = Uint8Array.from(BLOCKS, (b) => (b.solid ? 1 : 0));
const TRANSPARENT = Uint8Array.from(BLOCKS, (b) => (b.transparent ? 1 : 0));

export function getBlockDef(id: number): BlockDef {
  const def = BLOCKS[id];
  if (!def) throw new Error(`Unknown block id ${id}`);
  return def;
}

export function isSolid(id: number): boolean {
  return SOLID[id] === 1;
}

export function isTransparent(id: number): boolean {
  return TRANSPARENT[id] === 1;
}
