import { useEffect, useState } from "react";
import scanTile from "@/assets/scan-tile.jpg";
import type { Selection } from "./Urbaneuron";
import { CornerBrackets, LogoMark, StatusPill } from "./Chrome";
import { MODEL_META, SCAN_STEPS as STEPS } from "@/lib/model";

export function ScanView({ selection }: { selection: Selection }) {
  const [step, setStep] = useState(0);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const t = window.setInterval(() => setStep((s) => (s + 1) % STEPS.length), 480);
    const p = window.setInterval(
      () => setProgress((v) => Math.min(100, v + 2 + Math.random() * 3)),
      90,
    );
    return () => {
      window.clearInterval(t);
      window.clearInterval(p);
    };
  }, []);

  const aspect = selection.widthPx / selection.heightPx;
  const frameH = 560;
  const frameW = Math.round(frameH * aspect);

  return (
    <section
      className="relative min-h-screen w-full"
      style={{ animation: "vignette-in 600ms ease-out" }}
    >
      <div className="absolute inset-0" style={{
        background: "radial-gradient(ellipse at center, #0F1B31 0%, #06101E 60%, #030811 100%)",
      }} />
      <div className="pointer-events-none absolute inset-0" style={{
        backgroundImage:
          "linear-gradient(rgba(42,54,82,0.25) 1px, transparent 1px), linear-gradient(90deg, rgba(42,54,82,0.25) 1px, transparent 1px)",
        backgroundSize: "48px 48px",
        maskImage: "radial-gradient(ellipse at center, black 25%, transparent 75%)",
      }} />

      <div className="relative z-20 flex items-center justify-between p-6">
        <LogoMark />
        <StatusPill label="Session" live>SCAN-{Date.now().toString().slice(-6)}</StatusPill>
      </div>

      <div className="relative z-10 flex min-h-[calc(100vh-96px)] flex-col items-center justify-center px-4 pb-16">
        <div
          className="relative"
          style={{
            width: `min(90vw, ${frameW}px)`,
            aspectRatio: `${aspect}`,
            animation: "aperture-in 700ms cubic-bezier(0.22,1,0.36,1) both",
          }}
        >
          <div
            className="absolute -inset-px rounded-sm"
            style={{
              background:
                "linear-gradient(135deg, var(--signal) 0%, rgba(255,107,26,0.2) 30%, rgba(27,42,74,1) 60%, var(--signal) 100%)",
              padding: 1,
            }}
          >
            <div className="h-full w-full rounded-sm bg-navy-950" />
          </div>

          <div className="absolute inset-[1px] overflow-hidden rounded-sm">
            <img
              src={scanTile}
              alt="Satellite tile being segmented"
              width={1024}
              height={1024}
              className="h-full w-full object-cover"
              style={{
                filter: "saturate(0.85) contrast(1.05) hue-rotate(-8deg) brightness(0.85)",
              }}
            />

            {}
            <div
              className="pointer-events-none absolute inset-0"
              style={{
                backgroundImage:
                  "linear-gradient(rgba(255,107,26,0.18) 1px, transparent 1px), linear-gradient(90deg, rgba(255,107,26,0.18) 1px, transparent 1px)",
                backgroundSize: "32px 32px",
                animation: "grid-pulse 2.4s ease-in-out infinite",
                mixBlendMode: "screen",
              }}
            />
            <div className="pointer-events-none absolute inset-0 bg-navy-950/30 mix-blend-multiply" />

            {}
            <div className="pointer-events-none absolute inset-0 overflow-hidden">
              <div
                className="absolute left-0 right-0 h-[3px]"
                style={{
                  background:
                    "linear-gradient(180deg, transparent 0%, rgba(255,140,66,0.9) 50%, transparent 100%)",
                  boxShadow:
                    "0 0 24px 4px rgba(255,107,26,0.65), 0 0 60px 12px rgba(255,107,26,0.35)",
                  animation: "scanline-sweep 2.4s ease-in-out infinite",
                }}
              />
            </div>

            <CornerBrackets color="var(--signal)" size={22} thickness={2} inset={8} pulse />

            <div className="pointer-events-none absolute left-1/2 top-1/2 h-16 w-16 -translate-x-1/2 -translate-y-1/2">
              <div className="absolute left-1/2 top-0 h-full w-px -translate-x-1/2 bg-[color:var(--signal)]/70" />
              <div className="absolute top-1/2 left-0 h-px w-full -translate-y-1/2 bg-[color:var(--signal)]/70" />
              <div className="absolute left-1/2 top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border border-[color:var(--signal)]" />
            </div>
          </div>

          <div className="absolute -left-3 top-0 -translate-x-full font-mono-ui text-[10px] uppercase tracking-[0.2em] text-cool-grey">
            H&nbsp;{selection.heightPx}
          </div>
          <div className="absolute -top-6 left-0 font-mono-ui text-[10px] uppercase tracking-[0.2em] text-cool-grey">
            W&nbsp;{selection.widthPx}&nbsp;PX
          </div>
          <div className="absolute -bottom-6 right-0 font-mono-ui text-[10px] uppercase tracking-[0.2em] text-cool-grey">
            LAT {selection.lat.toFixed(4)} · LON {selection.lng.toFixed(4)}
          </div>
        </div>

        <div className="mt-16 w-full max-w-2xl">
          <div className="flex items-center justify-between font-mono-ui text-[11px] uppercase tracking-[0.24em]">
            <div className="text-[color:var(--signal)]">
              <span className="mr-2 inline-block h-1.5 w-1.5 translate-y-[-1px] rounded-full bg-[color:var(--signal)] signal-glow" />
              {STEPS[step]}
            </div>
            <div className="text-offwhite">{Math.floor(progress)}%</div>
          </div>
          <div className="mt-3 h-[3px] w-full overflow-hidden bg-navy-600/60">
            <div
              className="h-full bg-[color:var(--signal)]"
              style={{ width: `${progress}%`, boxShadow: "0 0 10px var(--signal)" }}
            />
          </div>
          <div className="mt-6 grid grid-cols-2 gap-x-6 gap-y-1.5 font-mono-ui text-[10px] uppercase tracking-[0.2em] text-cool-grey md:grid-cols-4">
            <div><span className="text-offwhite/80">MODEL</span> {MODEL_META.architecture}</div>
            <div><span className="text-offwhite/80">ENCODER</span> {MODEL_META.encoder}</div>
            <div><span className="text-offwhite/80">TILE</span> {MODEL_META.tile}</div>
            <div><span className="text-offwhite/80">GSD</span> {MODEL_META.gsd}</div>
            <div><span className="text-offwhite/80">TRAIN</span> {MODEL_META.dataset} · {MODEL_META.mix}</div>
            <div><span className="text-offwhite/80">PARAMS</span> {MODEL_META.params}</div>
            <div><span className="text-offwhite/80">BANDS</span> {MODEL_META.bands}</div>
            <div><span className="text-offwhite/80">SEL</span> {selection.widthPx}×{selection.heightPx} PX</div>
          </div>
          <div className="mt-3 font-mono-ui text-[10px] uppercase tracking-[0.2em] text-cool-grey">
            <span className="text-offwhite/80">INFERENCE</span> {MODEL_META.inference}
          </div>
        </div>
      </div>
    </section>
  );
}