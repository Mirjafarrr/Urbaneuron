
export const MODEL_META = {
  architecture: "U-Net++",
  encoder: "ResNeXt-50 (32×4d)",
  framework: "PyTorch",
  params: "~51.0M",
  dataset: "LoveDA",
  tile: "512×512",
  gsd: "0.30 M/PX",
  bands: "RGB",
  mix: "85/15 URBAN/RURAL",
  inference: "DOCKER · RTX 4060",
} as const;

export type LovedaClass = {
  index: number;
  label: string;
  color: string;
};

export const LOVEDA_CLASSES: LovedaClass[] = [
  { index: 2, label: "Building", color: "#FF6B1A" },
  { index: 3, label: "Road", color: "#7B8494" },
  { index: 4, label: "Water", color: "#5F8FC7" },
  { index: 6, label: "Forest", color: "#4FB3A9" },
  { index: 7, label: "Agriculture", color: "#C9B458" },
  { index: 5, label: "Barren", color: "#B08968" },
  { index: 1, label: "Background", color: "#3A4665" },
  { index: 0, label: "Ignore / No-Data", color: "#1B2A4A" },
];

export const DEMO_DISTRIBUTION: Record<string, number> = {
  Building: 38.4,
  Road: 16.1,
  Water: 6.7,
  Forest: 17.5,
  Agriculture: 12.2,
  Barren: 5.3,
  Background: 3.1,
  "Ignore / No-Data": 0.7,
};

export const SCAN_STEPS = [
  "TILING SELECTION → 512×512",
  "NORMALIZING RGB BANDS",
  "RESNEXT-50 (32×4D) ENCODER · STAGE 1-4",
  "NESTED DECODER · X0,1 → X0,2",
  "NESTED DECODER · X0,3 → X0,4",
  "PER-PIXEL LOGITS · 8 CHANNELS",
  "ARGMAX / CLASS ASSIGNMENT",
  "STITCHING TILE MOSAIC",
];
