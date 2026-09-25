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
}

interface BlockSpec {
  solid: boolean;
  transparent: boolean;
  emissive: number;
  /** One material for all faces, or per-face materials. */
  faces: MaterialName | { top: MaterialName; side: MaterialName; bottom: MaterialName } | null;
  footstep: FootstepMaterial | null;
}

const SPECS = {
  air: { solid: false, transparent: true, emissive: 0, faces: null, footstep: null },
  grass: {
    solid: true,
    transparent: false,
    emissive: 0,
    faces: { top: 'grass_top', side: 'grass_side', bottom: 'dirt' },
    footstep: 'grass',
  },
  dirt: { solid: true, transparent: false, emissive: 0, faces: 'dirt', footstep: 'grass' },
  stone: { solid: true, transparent: false, emissive: 0, faces: 'stone', footstep: 'stone' },
  cobblestone: { solid: true, transparent: false, emissive: 0, faces: 'cobblestone', footstep: 'stone' },
  gravel: { solid: true, transparent: false, emissive: 0, faces: 'gravel', footstep: 'gravel' },
  sand: { solid: true, transparent: false, emissive: 0, faces: 'sand', footstep: 'sand' },
  water: { solid: false, transparent: true, emissive: 0, faces: 'water', footstep: null },
  oak_log: {
    solid: true,
    transparent: false,
    emissive: 0,
    faces: { top: 'oak_log_top', side: 'oak_log_side', bottom: 'oak_log_top' },
    footstep: 'wood',
  },
  oak_leaves: { solid: true, transparent: true, emissive: 0, faces: 'oak_leaves', footstep: 'grass' },
  oak_planks: { solid: true, transparent: false, emissive: 0, faces: 'oak_planks', footstep: 'wood' },
  glass: { solid: true, transparent: true, emissive: 0, faces: 'glass', footstep: 'stone' },
  torch: { solid: false, transparent: true, emissive: 8, faces: 'torch', footstep: null },
  lava: { solid: false, transparent: false, emissive: 4, faces: 'lava', footstep: null },
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
