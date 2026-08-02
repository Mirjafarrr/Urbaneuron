import heroBg from "@/assets/hero-satellite.jpg";
import { CornerBrackets, LogoMark, SignalButton, StatusPill } from "./Chrome";

export function HeroView({ onStart }: { onStart: () => void }) {
  return (
    <section className="relative min-h-screen w-full overflow-hidden">
      {}
      <div className="absolute inset-0">
        <img
          src={heroBg}
          alt=""
          width={1920}
          height={1280}
          className="h-full w-full object-cover opacity-70"
          style={{ filter: "hue-rotate(200deg) saturate(0.7) contrast(1.05)" }}
        />
        <div className="absolute inset-0 bg-gradient-to-b from-navy-950/60 via-navy-950/70 to-navy-950" />
        <div className="absolute inset-0" style={{
          backgroundImage:
            "linear-gradient(rgba(42,54,82,0.35) 1px, transparent 1px), linear-gradient(90deg, rgba(42,54,82,0.35) 1px, transparent 1px)",
          backgroundSize: "64px 64px",
          maskImage: "radial-gradient(ellipse at center, black 20%, transparent 75%)",
          opacity: 0.35,
        }} />
      </div>

      {}
      <div className="relative z-20 flex items-center justify-between p-6 md:px-10">
        <LogoMark />
        <div className="hidden md:flex items-center gap-3">
          <StatusPill label="Node" live>EU-WEST-3</StatusPill>
          <StatusPill label="Model">USEG-v4.2</StatusPill>
        </div>
      </div>

      {}
      <div className="pointer-events-none absolute inset-x-0 top-1/2 -translate-y-1/2 z-10 flex items-center">
        <div className="h-px flex-1 bg-gradient-to-r from-transparent via-navy-600/60 to-transparent" />
      </div>

      <div className="relative z-20 mx-auto flex max-w-6xl flex-col items-start px-6 pt-16 pb-32 md:px-10 md:pt-24">
        <div className="mb-5 flex items-center gap-3 font-mono-ui text-[10px] uppercase tracking-[0.3em] text-cool-grey">
          <span className="h-1.5 w-1.5 rounded-full bg-[color:var(--signal)] signal-glow" />
          Orbital Segmentation Platform / v4.2
        </div>
        <h1 className="font-display text-[44px] leading-[1.02] font-semibold tracking-tight text-offwhite md:text-[76px]">
          Extract the structure<br />
          of any city block.
        </h1>
        <p className="mt-6 max-w-xl text-[15px] leading-relaxed text-offwhite/70">
          Urbaneuron runs a purpose-built segmentation model over urban satellite
          imagery. Draw a bounding box on the globe — up to 2560&nbsp;&times;&nbsp;2560&nbsp;px —
          and receive a pixel-accurate map of buildings, roads, vegetation and water.
        </p>

        <div className="mt-10 flex flex-wrap items-center gap-4">
          <SignalButton size="lg" onClick={onStart}>
            Open Analysis Console
            <span aria-hidden className="ml-1">→</span>
          </SignalButton>
          <div className="font-mono-ui text-[11px] uppercase tracking-[0.22em] text-cool-grey">
            [ Ctrl + Drag to select region ]
          </div>
        </div>

        {}
        <div className="mt-20 grid w-full grid-cols-2 gap-px overflow-hidden rounded-sm border border-navy-600/70 bg-navy-600/30 md:grid-cols-4">
          {[
            ["MAX RESOLUTION", "2560 × 2560", "px"],
            ["CLASSES", "07", "semantic"],
            ["PARAMETERS", "51.0", "M"],
            ["LATENCY", "4.1", "s / tile"],
          ].map(([k, v, u]) => (
            <div key={k} className="glass-panel !rounded-none px-5 py-5">
              <div className="font-mono-ui text-[9px] uppercase tracking-[0.28em] text-cool-grey">{k}</div>
              <div className="mt-2 flex items-baseline gap-1.5">
                <div className="font-mono-ui text-2xl text-offwhite">{v}</div>
                <div className="font-mono-ui text-[10px] uppercase tracking-[0.2em] text-cool-grey">{u}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {}
      <div className="pointer-events-none absolute inset-6 md:inset-10">
        <CornerBrackets color="rgba(90,100,114,0.6)" size={22} thickness={1} />
      </div>

      {}
      <div className="absolute bottom-0 inset-x-0 z-20 border-t border-navy-600/70 bg-navy-950/70 backdrop-blur">
        <div className="overflow-hidden">
          <div className="flex whitespace-nowrap py-2 font-mono-ui text-[10px] uppercase tracking-[0.28em] text-cool-grey"
               style={{ animation: "ticker 60s linear infinite" }}>
            {Array.from({ length: 2 }).map((_, i) => (
              <div key={i} className="flex shrink-0 items-center gap-8 px-8">
                <span><span className="text-offwhite/70">LAT</span> 48.8566</span>
                <span><span className="text-offwhite/70">LON</span> 2.3522</span>
                <span className="text-[color:var(--signal)]">● TILE STREAM NOMINAL</span>
                <span><span className="text-offwhite/70">CONSTELLATION</span> USEG-A / USEG-B</span>
                <span><span className="text-offwhite/70">GSD</span> 0.30 m/px</span>
                <span><span className="text-offwhite/70">CLOUD COVER</span> 03%</span>
                <span><span className="text-offwhite/70">PASS</span> #48127</span>
                <span className="text-[color:var(--signal)]">● QUEUE 0</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}