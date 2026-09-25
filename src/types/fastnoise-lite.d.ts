// fastnoise-lite ships plain JS without typings; only the API we use is declared.
declare module 'fastnoise-lite' {
  type NoiseType = 'OpenSimplex2' | 'OpenSimplex2S' | 'Cellular' | 'Perlin' | 'ValueCubic' | 'Value';
  type FractalType = 'None' | 'FBm' | 'Ridged' | 'PingPong' | 'DomainWarpProgressive' | 'DomainWarpIndependent';
  type RotationType3D = 'None' | 'ImproveXYPlanes' | 'ImproveXZPlanes';

  export default class FastNoiseLite {
    static readonly NoiseType: { readonly [K in NoiseType]: K };
    static readonly FractalType: { readonly [K in FractalType]: K };
    static readonly RotationType3D: { readonly [K in RotationType3D]: K };

    constructor(seed?: number);
    SetSeed(seed: number): void;
    SetFrequency(frequency: number): void;
    SetNoiseType(noiseType: NoiseType): void;
    SetRotationType3D(rotationType: RotationType3D): void;
    SetFractalType(fractalType: FractalType): void;
    SetFractalOctaves(octaves: number): void;
    SetFractalLacunarity(lacunarity: number): void;
    SetFractalGain(gain: number): void;
    SetFractalWeightedStrength(weightedStrength: number): void;
    /** Must be called with exactly 2 or 3 arguments (dispatches on arguments.length). */
    GetNoise(x: number, y: number): number;
    GetNoise(x: number, y: number, z: number): number;
  }
}
