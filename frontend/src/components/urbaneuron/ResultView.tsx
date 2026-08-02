import { useRef, useState, useEffect } from "react";
import scanTile from "@/assets/scan-tile.jpg";
import scanSeg from "@/assets/scan-tile-segmented.jpg";
import type { Selection } from "./Urbaneuron";
import { CornerBrackets, LogoMark, SignalButton, StatusPill } from "./Chrome";
import { DEMO_DISTRIBUTION, LOVEDA_CLASSES, MODEL_META } from "@/lib/model";
import type { SegmentResult } from "@/lib/inference.functions";

function base64ToBlob(b64: string, mime: string): Blob {
  const raw = atob(b64);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return new Blob([bytes], { type: mime });
}

function dataUriToBlob(uri: string): { blob: Blob; mime: string } | null {
  const m = uri.match(/^data:([^;]+);base64,(.+)$/);
  if (!m) return null;
  return { mime: m[1], blob: base64ToBlob(m[2], m[1]) };
}

function triggerDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export function ResultView({
  selection,
  result,
  onReset,
  onHome,
}: {
  selection: Selection;
  result?: SegmentResult | null;
  onReset: () => void;
  onHome: () => void;
}) {
  const [split, setSplit] = useState(55);
  const [exportOpen, setExportOpen] = useState(false);
  const [exporting, setExporting] = useState(false);
  const wrap = useRef<HTMLDivElement | null>(null);
  const exportMenuRef = useRef<HTMLDivElement | null>(null);
  const dragging = useRef(false);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (
        exportMenuRef.current &&
        !exportMenuRef.current.contains(e.target as Node)
      ) {
        setExportOpen(false);
      }
    }
    if (exportOpen) document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [exportOpen]);

  const dist = result?.distribution ?? DEMO_DISTRIBUTION;
  const legend = LOVEDA_CLASSES.map((c) => ({
    ...c,
    pct: dist[c.label] ?? 0,
  })).filter((c) => c.pct > 0);
  const maskSrc = result?.maskUrl ?? scanSeg;
  const overlaySrc = result?.overlayUrl;
  const originalSrc = result?.originalUrl;
  const live = result?.status === "live";
  const confidence = result?.meanConfidence;

  const setFromEvent = (clientX: number) => {
    if (!wrap.current) return;
    const r = wrap.current.getBoundingClientRect();
    const v = ((clientX - r.left) / r.width) * 100;
    setSplit(Math.min(98, Math.max(2, v)));
  };

  const exportOriginal = async () => {
    setExportOpen(false);
    const src = originalSrc ?? scanTile;
    if (src.startsWith("data:")) {
      const parsed = dataUriToBlob(src);
      if (parsed) triggerDownload(parsed.blob, "urbaneuron_original.png");
    } else {
      setExporting(true);
      try {
        const r = await fetch(src);
        const blob = await r.blob();
        triggerDownload(blob, "urbaneuron_original.png");
      } catch {
      }
      setExporting(false);
    }
  };

  const exportSegmented = () => {
    setExportOpen(false);
    const src = overlaySrc ?? maskSrc;
    if (src.startsWith("data:")) {
      const parsed = dataUriToBlob(src);
      if (parsed) triggerDownload(parsed.blob, "urbaneuron_segmented.png");
    }
  };

  const exportBoth = async () => {
    setExportOpen(false);
    setExporting(true);
    const { default: JSZip } = await import("jszip");
    const zip = new JSZip();

    const addFromUrl = async (
      label: string,
      src: string | undefined,
      fallback: string,
    ) => {
      if (src) {
        if (src.startsWith("data:")) {
          const parsed = dataUriToBlob(src);
          if (parsed) zip.file(label, parsed.blob);
        } else {
          try {
            const r = await fetch(src);
            const blob = await r.blob();
            zip.file(label, blob);
          } catch {
          }
        }
      } else if (fallback.startsWith("data:")) {
        const parsed = dataUriToBlob(fallback);
        if (parsed) zip.file(label, parsed.blob);
      }
    };

    await addFromUrl("original.png", originalSrc, scanTile);
    await addFromUrl("segmented.png", overlaySrc ?? undefined, scanSeg);

    const zipBlob = await zip.generateAsync({ type: "blob" });
    triggerDownload(zipBlob, "urbaneuron_result.zip");
    setExporting(false);
  };

  return (
    <section
      className="relative min-h-screen w-full"
      style={{ animation: "vignette-in 500ms ease-out" }}
    >
      <div
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse at top, #14213D 0%, #0B1524 60%, #06101E 100%)",
        }}
      />

      <div className="relative z-10 mx-auto flex min-h-screen max-w-7xl flex-col gap-6 px-4 py-6 md:px-8 md:py-8">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <LogoMark />
          <StatusPill label="Session" live>
            SEGMENT COMPLETE
          </StatusPill>
          <StatusPill label="Inference" live={live}>
            {live ? "MODAL · LIVE" : (result?.note ?? "SIMULATED")}
          </StatusPill>
        </div>

        <div className="grid flex-1 grid-cols-1 gap-6 lg:grid-cols-[1fr_320px]">
          <div className="flex flex-col gap-3">
            <div className="flex items-baseline justify-between">
              <div>
                <div className="font-mono-ui text-[10px] uppercase tracking-[0.28em] text-[color:var(--signal)]">
                  ● OUTPUT READY
                </div>
                <h2 className="font-display text-3xl font-semibold tracking-tight text-offwhite md:text-4xl">
                  Segmentation Result
                </h2>
              </div>
              <div className="hidden md:block font-mono-ui text-[10px] uppercase tracking-[0.22em] text-cool-grey">
                DRAG HANDLE TO COMPARE
              </div>
            </div>

            <div
              ref={wrap}
              className="relative aspect-square w-full select-none overflow-hidden rounded-sm"
              onPointerMove={(e) =>
                dragging.current && setFromEvent(e.clientX)
              }
              onPointerUp={() => (dragging.current = false)}
              onPointerLeave={() => (dragging.current = false)}
            >
              <div
                className="pointer-events-none absolute -inset-px rounded-sm"
                style={{
                  background:
                    "linear-gradient(135deg, var(--signal) 0%, rgba(27,42,74,1) 40%, rgba(27,42,74,1) 60%, var(--signal) 100%)",
                }}
              />
              <div className="absolute inset-[1px] overflow-hidden rounded-sm bg-navy-950">
                <img
                  src={originalSrc ?? scanTile}
                  alt="Raw satellite tile"
                  width={1024}
                  height={1024}
                  loading="lazy"
                  className="absolute inset-0 h-full w-full object-cover"
                />
                <div
                  className="absolute inset-0 overflow-hidden"
                  style={{ clipPath: `inset(0 0 0 ${split}%)` }}
                >
                  <img
                    src={overlaySrc ?? maskSrc}
                    alt="AI segmentation overlay"
                    width={1024}
                    height={1024}
                    loading="lazy"
                    className="h-full w-full object-cover"
                  />
                </div>
                <div
                  className="absolute inset-y-0"
                  style={{
                    left: `${split}%`,
                    transform: "translateX(-50%)",
                  }}
                >
                  <div className="h-full w-px bg-[color:var(--signal)] signal-glow" />
                  <button
                    onPointerDown={(e) => {
                      dragging.current = true;
                      (e.currentTarget as HTMLElement).setPointerCapture(
                        e.pointerId,
                      );
                    }}
                    className="absolute left-1/2 top-1/2 flex h-10 w-10 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border border-[color:var(--signal)] bg-navy-900/80 font-mono-ui text-xs text-[color:var(--signal)] hover:signal-glow"
                    aria-label="Drag to compare"
                  >
                    ‹›
                  </button>
                </div>
                <div className="absolute left-3 top-3 glass-panel rounded-sm px-2 py-1 font-mono-ui text-[10px] uppercase tracking-[0.22em] text-offwhite/80">
                  RAW / RGB
                </div>
                <div className="absolute right-3 top-3 glass-panel rounded-sm px-2 py-1 font-mono-ui text-[10px] uppercase tracking-[0.22em] text-[color:var(--signal)]">
                  SEGMENTED
                </div>
                <CornerBrackets
                  color="var(--signal)"
                  size={18}
                  thickness={2}
                  inset={6}
                />
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-x-6 gap-y-2 font-mono-ui text-[10px] uppercase tracking-[0.22em] text-cool-grey">
              <div>
                <span className="text-offwhite/80">TILE</span>{" "}
                {selection.widthPx}×{selection.heightPx} PX
              </div>
              <div>
                <span className="text-offwhite/80">LAT/LON</span>{" "}
                {selection.lat.toFixed(4)}, {selection.lng.toFixed(4)}
              </div>
              <div>
                <span className="text-offwhite/80">ZOOM</span>{" "}
                {selection.zoom.toFixed(2)}
              </div>
              <div>
                <span className="text-offwhite/80">MODEL</span>{" "}
                {MODEL_META.architecture} / {MODEL_META.encoder}
              </div>
              <div>
                <span className="text-offwhite/80">TRAIN</span>{" "}
                {MODEL_META.dataset} · {MODEL_META.mix}
              </div>
              {typeof confidence === "number" && (
                <div className="text-[color:var(--signal)]">
                  MEAN CONF · {confidence.toFixed(3)}
                </div>
              )}
              {typeof result?.durationMs === "number" && (
                <div>
                  <span className="text-offwhite/80">LATENCY</span>{" "}
                  {result.durationMs} MS
                </div>
              )}
            </div>
          </div>

          <aside className="glass-panel flex flex-col gap-5 rounded-sm p-5">
            <div>
              <div className="font-mono-ui text-[10px] uppercase tracking-[0.28em] text-cool-grey">
                CLASS DISTRIBUTION · LOVEDA
              </div>
              <div className="mt-3 flex h-2 w-full overflow-hidden rounded-sm">
                {legend.map((c) => (
                  <div
                    key={c.label}
                    style={{ background: c.color, width: `${c.pct}%` }}
                  />
                ))}
              </div>
            </div>
            <div className="flex flex-col divide-y divide-navy-600/70">
              {legend.map((c) => (
                <div
                  key={c.label}
                  className="flex items-center justify-between py-2.5"
                >
                  <div className="flex items-center gap-3">
                    <span
                      className="h-3 w-3 rounded-sm"
                      style={{ background: c.color }}
                    />
                    <span className="text-[13px] text-offwhite">
                      <span className="font-mono-ui text-[11px] text-cool-grey">
                        {c.index}
                      </span>{" "}
                      {c.label}
                    </span>
                  </div>
                  <span className="font-mono-ui text-[12px] text-cool-grey">
                    {c.pct.toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>

            <div className="mt-auto flex flex-col gap-2">
              <SignalButton variant="ghost" onClick={onReset}>
                ← Select Another Area
              </SignalButton>
              <SignalButton variant="ghost" onClick={onHome}>
                ← Home
              </SignalButton>

              {}
              <div className="relative" ref={exportMenuRef}>
                <SignalButton
                  disabled={exporting}
                  onClick={() => setExportOpen((o) => !o)}
                  className="w-full"
                >
                  {exporting ? "Packaging…" : "⬇ Export"}
                </SignalButton>
                {exportOpen && (
                  <div className="absolute -top-2 left-0 w-full -translate-y-full rounded-sm border border-navy-600/70 bg-navy-900 py-1 shadow-lg">
                    <button
                      onClick={exportOriginal}
                      className="flex w-full items-center gap-2 px-3 py-2 font-mono-ui text-[10px] uppercase tracking-[0.18em] text-offwhite/80 hover:bg-[color:var(--signal)]/10"
                    >
                      <span className="text-sm">🖼</span> Original Image
                    </button>
                    <button
                      onClick={exportSegmented}
                      className="flex w-full items-center gap-2 px-3 py-2 font-mono-ui text-[10px] uppercase tracking-[0.18em] text-offwhite/80 hover:bg-[color:var(--signal)]/10"
                    >
                      <span className="text-sm">🎨</span> Segmented Overlay
                    </button>
                    <button
                      onClick={exportBoth}
                      className="flex w-full items-center gap-2 px-3 py-2 font-mono-ui text-[10px] uppercase tracking-[0.18em] text-[color:var(--signal)] hover:bg-[color:var(--signal)]/10"
                    >
                      <span className="text-sm">📦</span> Both as .zip
                    </button>
                  </div>
                )}
              </div>
            </div>
          </aside>
        </div>
      </div>
    </section>
  );
}