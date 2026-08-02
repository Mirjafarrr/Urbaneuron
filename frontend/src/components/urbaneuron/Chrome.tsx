import type { ReactNode } from "react";

export function LogoMark({ className = "" }: { className?: string }) {
  return (
    <div className={`flex items-center gap-2.5 ${className}`}>
      <div className="relative h-7 w-7">
        <div className="absolute inset-0 rounded-sm border border-[color:var(--signal)] opacity-90" />
        <div className="absolute inset-1 rounded-sm bg-[color:var(--signal)]/20" />
        <div className="absolute left-1/2 top-0 h-full w-px -translate-x-1/2 bg-[color:var(--signal)]/60" />
        <div className="absolute top-1/2 left-0 h-px w-full -translate-y-1/2 bg-[color:var(--signal)]/60" />
      </div>
      <div className="leading-none">
        <div className="font-display text-[15px] font-semibold tracking-tight text-offwhite">
          URBANEURON
        </div>
        <div className="font-mono-ui text-[9px] uppercase tracking-[0.25em] text-cool-grey">
          Segmentation OPS
        </div>
      </div>
    </div>
  );
}

export function StatusPill({
  label,
  live = false,
  children,
}: {
  label: string;
  live?: boolean;
  children?: ReactNode;
}) {
  return (
    <div className="glass-panel flex items-center gap-2 rounded-full px-3 py-1.5 font-mono-ui text-[10px] uppercase tracking-[0.18em] text-offwhite/80">
      <span
        className={`h-1.5 w-1.5 rounded-full ${
          live ? "bg-[color:var(--signal)] signal-glow" : "bg-cool-grey"
        }`}
      />
      <span className="text-cool-grey">{label}</span>
      {children && <span className="text-offwhite">{children}</span>}
    </div>
  );
}

export function TopBar({ right }: { right?: ReactNode }) {
  return (
    <div className="pointer-events-none absolute inset-x-0 top-0 z-30 flex items-start justify-between p-4 md:p-6">
      <div className="pointer-events-auto">
        <LogoMark />
      </div>
      <div className="pointer-events-auto flex items-center gap-2">{right}</div>
    </div>
  );
}

export function CornerBrackets({
  color = "var(--signal)",
  size = 18,
  thickness = 2,
  inset = 0,
  pulse = false,
}: {
  color?: string;
  size?: number;
  thickness?: number;
  inset?: number;
  pulse?: boolean;
}) {
  const style = {
    "--bc": color,
    "--bs": `${size}px`,
    "--bt": `${thickness}px`,
    "--bi": `${inset}px`,
  } as React.CSSProperties;
  const anim = pulse ? { animation: "bracket-pulse 1.6s ease-in-out infinite" } : undefined;
  const base =
    "pointer-events-none absolute w-[var(--bs)] h-[var(--bs)] border-[color:var(--bc)]";
  return (
    <div className="pointer-events-none absolute inset-0" style={style}>
      <div
        className={`${base} border-l-[length:var(--bt)] border-t-[length:var(--bt)]`}
        style={{ top: "var(--bi)", left: "var(--bi)", ...anim }}
      />
      <div
        className={`${base} border-r-[length:var(--bt)] border-t-[length:var(--bt)]`}
        style={{ top: "var(--bi)", right: "var(--bi)", ...anim }}
      />
      <div
        className={`${base} border-l-[length:var(--bt)] border-b-[length:var(--bt)]`}
        style={{ bottom: "var(--bi)", left: "var(--bi)", ...anim }}
      />
      <div
        className={`${base} border-r-[length:var(--bt)] border-b-[length:var(--bt)]`}
        style={{ bottom: "var(--bi)", right: "var(--bi)", ...anim }}
      />
    </div>
  );
}

export function SignalButton({
  children,
  onClick,
  disabled,
  variant = "primary",
  className = "",
  size = "md",
  type = "button",
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "primary" | "ghost";
  className?: string;
  size?: "sm" | "md" | "lg";
  type?: "button" | "submit";
}) {
  const sizes = {
    sm: "h-9 px-4 text-[11px]",
    md: "h-11 px-5 text-xs",
    lg: "h-13 px-7 text-sm",
  }[size];
  const base =
    "group relative inline-flex items-center justify-center gap-2 rounded-sm font-mono-ui uppercase tracking-[0.22em] transition-all duration-200 disabled:cursor-not-allowed disabled:opacity-40";
  const styles =
    variant === "primary"
      ? "bg-[color:var(--signal)] text-navy-950 hover:bg-[color:var(--signal-bright)] hover:signal-glow"
      : "border border-navy-600 text-offwhite/85 hover:border-[color:var(--signal)] hover:text-offwhite bg-navy-800/40";
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`${base} ${sizes} ${styles} ${className}`}
    >
      {children}
    </button>
  );
}