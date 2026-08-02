import { z } from "zod";

const InputSchema = z.object({
  widthPx: z.number().int().min(1).max(2560),
  heightPx: z.number().int().min(1).max(2560),
  lng: z.number().min(-180).max(180),
  lat: z.number().min(-90).max(90),
  zoom: z.number().min(0).max(24),
});

export type SegmentResult = {
  status: "live" | "simulated";
  note?: string;
  maskUrl?: string;
  overlayUrl?: string;
  originalUrl?: string;
  meanConfidence?: number;
  distribution?: Record<string, number>;
  durationMs?: number;
};

export async function segmentTile(
  input: z.infer<typeof InputSchema>,
): Promise<SegmentResult> {
  const parsed = InputSchema.safeParse(input);
  if (!parsed.success) {
    return { status: "simulated", note: `INVALID INPUT: ${parsed.error.message}` };
  }

  const { data } = parsed;
  const endpoint =
    import.meta.env["VITE_INFERENCE_API_URL"] ?? "http:

  const started = Date.now();

  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        lat: data.lat,
        lng: data.lng,
        zoom: data.zoom,
        width_px: data.widthPx,
        height_px: data.heightPx,
        tile_size: 512,
      }),
    });

    if (!res.ok) {
      return { status: "simulated", note: `INFERENCE HTTP ${res.status}` };
    }

    const body = (await res.json()) as {
      mask_url?: string;
      maskUrl?: string;
      overlay_url?: string;
      overlayUrl?: string;
      original_url?: string;
      originalUrl?: string;
      mean_confidence?: number;
      distribution?: Record<string, number>;
    };

    return {
      status: "live",
      maskUrl: body.mask_url ?? body.maskUrl,
      overlayUrl: body.overlay_url ?? body.overlayUrl,
      originalUrl: body.original_url ?? body.originalUrl,
      meanConfidence: body.mean_confidence,
      distribution: body.distribution,
      durationMs: Date.now() - started,
    };
  } catch {
    return { status: "simulated", note: "INFERENCE ENDPOINT UNREACHABLE" };
  }
}