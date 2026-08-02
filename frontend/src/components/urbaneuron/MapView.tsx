import "mapbox-gl/dist/mapbox-gl.css";

import { useEffect, useMemo, useRef, useState } from "react";
import type { Selection } from "./Urbaneuron";
import { CornerBrackets, LogoMark, SignalButton, StatusPill } from "./Chrome";

const MAX_PX = 2560;
const FIXED_ZOOM = 19;

type Box = { x: number; y: number; w: number; h: number };

type Coord = { lng: number; lat: number };

const PRESETS: { label: string; coord: Coord }[] = [
  { label: "Paris, FR", coord: { lng: 2.3522, lat: 48.8566 } },
  { label: "Manhattan, NY", coord: { lng: -73.9857, lat: 40.758 } },
  { label: "Tokyo, JP", coord: { lng: 139.7671, lat: 35.6812 } },
  { label: "Barcelona, ES", coord: { lng: 2.1734, lat: 41.3851 } },
  { label: "Cairo, EG", coord: { lng: 31.2357, lat: 30.0444 } },
];

type GeocodingFeature = {
  place_name: string;
  center: [number, number];
};

export function MapView({
  onAnalyze,
  onBack,
  transitioning,
}: {
  onAnalyze: (sel: Selection) => void;
  onBack: () => void;
  transitioning: boolean;
}) {
  const mapContainer = useRef<HTMLDivElement | null>(null);
  const overlayRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<mapboxgl.Map | null>(null);

  const [box, setBox] = useState<Box | null>(null);
  const [drawing, setDrawing] = useState(false);
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null);
  const [query, setQuery] = useState("");
  const [suggestOpen, setSuggestOpen] = useState(false);
  const [center, setCenter] = useState<Coord>(PRESETS[0].coord);
  const [zoom, setZoom] = useState(FIXED_ZOOM);
  const [suggestions, setSuggestions] = useState<GeocodingFeature[]>([]);
  const [searching, setSearching] = useState(false);
  const [mapLoading, setMapLoading] = useState(true);
  const [mapError, setMapError] = useState<string | null>(null);
  const dragStart = useRef<{ x: number; y: number } | null>(null);
  const geocodeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const MAPBOX_TOKEN = import.meta.env["VITE_MAPBOX_TOKEN"] as string | undefined;

  useEffect(() => {
    let mounted = true;
    if (mapRef.current) {
      mapRef.current.remove();
      mapRef.current = null;
    }

    (async () => {
      const mapboxgl = (await import("mapbox-gl")).default;
      if (!mounted || !mapContainer.current) return;

      if (!MAPBOX_TOKEN) {
        setMapError("Mapbox token not set — add VITE_MAPBOX_TOKEN to .env");
        setMapLoading(false);
        return;
      }

      mapboxgl.accessToken = MAPBOX_TOKEN;

      const map = new mapboxgl.Map({
        container: mapContainer.current,
        style: "mapbox:
        center: [center.lng, center.lat],
        zoom: FIXED_ZOOM,
        attributionControl: false,
        pitchWithRotate: false,
        dragRotate: false,
        minZoom: 15,
        maxZoom: 21,
      });

      map.on("load", () => {
        if (mounted) setMapLoading(false);
      });

      map.on("error", (e) => {
        const msg = e.error?.message ?? "Unknown map error";
        console.error("[MapView] Mapbox error:", msg);
        setMapError(msg);
        setMapLoading(false);
      });

      map.on("move", () => {
        const c = map.getCenter();
        setCenter({ lng: c.lng, lat: c.lat });
        setZoom(map.getZoom());
      });
      mapRef.current = map;
    })();
    return () => {
      mounted = false;
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, []);

  const geocode = (q: string) => {
    if (!MAPBOX_TOKEN || q.trim().length === 0) return;
    setSearching(true);
    fetch(
      `https:
    )
      .then((r) => r.json())
      .then((data) => {
        setSuggestions((data.features as GeocodingFeature[]) ?? []);
      })
      .catch(() => setSuggestions([]))
      .finally(() => setSearching(false));
  };

  const onQueryChange = (val: string) => {
    setQuery(val);
    setSuggestOpen(true);
    if (val.trim().length > 0) {
      if (geocodeTimer.current) clearTimeout(geocodeTimer.current);
      geocodeTimer.current = setTimeout(() => geocode(val), 300);
    } else {
      setSuggestions([]);
    }
  };

  const localXY = (e: React.PointerEvent | PointerEvent) => {
    const r = overlayRef.current!.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };

  const clamp = (b: Box): Box => {
    const r = overlayRef.current!.getBoundingClientRect();
    let { x, y, w, h } = b;
    if (x < 0) {
      w += x;
      x = 0;
    }
    if (y < 0) {
      h += y;
      y = 0;
    }
    if (x + w > r.width) w = r.width - x;
    if (y + h > r.height) h = r.height - y;
    return { x, y, w: Math.max(0, w), h: Math.max(0, h) };
  };

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    const p = localXY(e);
    dragStart.current = p;
    setBox({ x: p.x, y: p.y, w: 0, h: 0 });
    setDrawing(true);
    setCursor(p);
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const p = localXY(e);
    setCursor(p);
    if (!drawing || !dragStart.current) return;
    const s = dragStart.current;
    const raw: Box = {
      x: Math.min(s.x, p.x),
      y: Math.min(s.y, p.y),
      w: Math.abs(p.x - s.x),
      h: Math.abs(p.y - s.y),
    };
    let clamped = clamp(raw);
    if (clamped.w > MAX_PX) clamped.w = MAX_PX;
    if (clamped.h > MAX_PX) clamped.h = MAX_PX;
    setBox(clamped);
  };
  const onPointerUp = (e: React.PointerEvent) => {
    (e.currentTarget as HTMLElement).releasePointerCapture?.(e.pointerId);
    setDrawing(false);
    dragStart.current = null;
  };

  const overSize = box && (box.w > MAX_PX || box.h > MAX_PX);
  const validSize = box && box.w >= 40 && box.h >= 40 && !overSize;

  const flyTo = (c: Coord) => {
    setQuery("");
    setSuggestOpen(false);
    setSuggestions([]);
    mapRef.current?.flyTo({
      center: [c.lng, c.lat],
      zoom: FIXED_ZOOM,
      duration: 1400,
    });
  };

  
  const filteredPresets = useMemo(
    () =>
      query.trim().length === 0
        ? PRESETS
        : PRESETS.filter((p) =>
            p.label.toLowerCase().includes(query.toLowerCase()),
          ),
    [query],
  );

  const submitQuery = () => {
    const m = query.match(
      /^\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*$/,
    );
    if (m) {
      const lat = parseFloat(m[1]);
      const lng = parseFloat(m[2]);
      if (Math.abs(lat) <= 90 && Math.abs(lng) <= 180) {
        flyTo({ lat, lng });
        return;
      }
    }
    if (suggestions[0]) {
      const [lng, lat] = suggestions[0].center;
      flyTo({ lat, lng });
      return;
    }
    if (filteredPresets[0]) flyTo(filteredPresets[0].coord);
  };

  return (
    <section
      className={`relative min-h-screen w-full transition-all duration-700 ${
        transitioning ? "" : ""
      }`}
    >
      {}
      <div ref={mapContainer} className="absolute inset-0" />

      {}
      {(mapLoading || mapError) && (
        <div className="absolute inset-0 z-5 flex items-center justify-center bg-navy-950/80">
          <div className="glass-panel rounded-sm px-6 py-4 text-center">
            {mapLoading ? (
              <>
                <span className="inline-block h-3 w-3 animate-pulse rounded-full bg-[color:var(--signal)]" />
                <p className="mt-2 font-mono-ui text-[10px] uppercase tracking-[0.22em] text-offwhite/70">
                  Loading satellite tiles…
                </p>
              </>
            ) : (
              <p className="font-mono-ui text-[11px] leading-relaxed text-[color:var(--danger)]">
                {mapError}
              </p>
            )}
          </div>
        </div>
      )}

      {}
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-navy-950/15 via-transparent to-navy-950/25" />

      {}
      {transitioning && box && <TransitionOverlay box={box} />}

      {}
      <div
        ref={overlayRef}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={() => setCursor(null)}
        className={`absolute inset-0 z-10 ${
          transitioning ? "pointer-events-none" : ""
        }`}
        style={{ cursor: "crosshair" }}
      >
        {}
        <CornerBrackets
          color="rgba(90,100,114,0.55)"
          size={24}
          thickness={1}
          inset={16}
        />

        {}
        {cursor && !box && !drawing && (
          <div
            className="pointer-events-none absolute font-mono-ui text-[10px] uppercase tracking-[0.2em] text-offwhite/70"
            style={{ left: cursor.x + 14, top: cursor.y + 14 }}
          >
            <div className="glass-panel rounded-sm px-2 py-1">
              CLICK-DRAG TO SELECT
            </div>
          </div>
        )}

        {}
        {box && box.w > 2 && box.h > 2 && (
          <SelectionBox
            box={box}
            over={!!overSize}
            drawing={drawing}
            cursor={cursor}
          />
        )}
      </div>

      {}
      <div className="pointer-events-none absolute inset-x-0 top-0 z-20 flex flex-wrap items-start justify-between gap-3 p-4 md:p-6">
        <div className="pointer-events-auto flex items-center gap-3">
          <button
            onClick={onBack}
            className="glass-panel flex h-9 items-center gap-2 rounded-sm px-3 font-mono-ui text-[10px] uppercase tracking-[0.22em] text-offwhite/80 hover:text-offwhite"
          >
            ← Home
          </button>
          <LogoMark />
        </div>

        {}
        <div className="pointer-events-auto relative w-full max-w-xl md:w-[520px]">
          <div className="glass-panel flex items-center rounded-sm px-3 focus-within:border-[color:var(--signal)] focus-within:signal-glow">
            <span className="font-mono-ui text-[10px] uppercase tracking-[0.24em] text-cool-grey">
              QRY&nbsp;›
            </span>
            <input
              value={query}
              onFocus={() => setSuggestOpen(true)}
              onBlur={() => setTimeout(() => setSuggestOpen(false), 200)}
              onChange={(e) => onQueryChange(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && submitQuery()}
              placeholder="Search city or paste lat, lng"
              className="h-11 w-full bg-transparent px-3 font-mono-ui text-[13px] text-offwhite placeholder:text-cool-grey/70 outline-none"
            />
            <div className="flex items-center gap-2 font-mono-ui text-[10px] uppercase tracking-[0.22em] text-cool-grey">
              {searching && (
                <span className="animate-pulse text-[color:var(--signal)]">
                  ●
                </span>
              )}
              <span>⏎ LOCATE</span>
            </div>
          </div>
          {suggestOpen && (filteredPresets.length > 0 || suggestions.length > 0) && (
            <div className="glass-panel absolute mt-1.5 w-full overflow-hidden rounded-sm">
              {}
              {filteredPresets.map((p) => (
                <button
                  key={p.label}
                  onMouseDown={() => flyTo(p.coord)}
                  className="flex w-full items-center justify-between border-b border-navy-600/60 px-3 py-2.5 text-left last:border-b-0 hover:bg-[color:var(--signal)]/10"
                >
                  <span className="text-[13px] text-offwhite">{p.label}</span>
                  <span className="font-mono-ui text-[10px] uppercase tracking-[0.2em] text-cool-grey">
                    {p.coord.lat.toFixed(4)}, {p.coord.lng.toFixed(4)}
                  </span>
                </button>
              ))}
              {}
              {suggestions.map((f, i) => (
                <button
                  key={f.place_name + i}
                  onMouseDown={() => {
                    const [lng, lat] = f.center;
                    flyTo({ lat, lng });
                  }}
                  className="flex w-full items-center justify-between border-b border-navy-600/60 px-3 py-2.5 text-left last:border-b-0 hover:bg-[color:var(--signal)]/10"
                >
                  <span className="text-[13px] text-offwhite">
                    {f.place_name}
                  </span>
                  <span className="font-mono-ui text-[10px] uppercase tracking-[0.2em] text-cool-grey">
                    GEO
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="pointer-events-auto hidden md:flex items-center gap-2">
          <StatusPill label="Node" live>
            EU-WEST-3
          </StatusPill>
        </div>
      </div>

      {}
      <div className="pointer-events-none absolute inset-x-0 bottom-0 z-20 p-4 md:p-6">
        <div className="pointer-events-auto glass-panel flex flex-wrap items-center justify-between gap-4 rounded-sm px-4 py-3 md:px-5 md:py-4">
          <div className="flex flex-wrap items-center gap-x-6 gap-y-2 font-mono-ui text-[10px] uppercase tracking-[0.22em] text-cool-grey">
            <div>
              <span className="text-offwhite/70">CENTER</span>{" "}
              <span className="text-offwhite">
                {center.lat.toFixed(4)}, {center.lng.toFixed(4)}
              </span>
            </div>
            <div>
              <span className="text-offwhite/70">ZOOM</span>{" "}
              <span className="text-offwhite">{zoom.toFixed(2)}</span>
            </div>
            <div>
              <span className="text-offwhite/70">SEL</span>{" "}
              <span
                className={
                  overSize ? "text-[color:var(--danger)]" : "text-offwhite"
                }
              >
                {box
                  ? `${Math.round(box.w)} × ${Math.round(box.h)} PX`
                  : "—"}
              </span>
            </div>
            <div>
              <span className="text-offwhite/70">LIMIT</span>{" "}
              <span className="text-offwhite">2560 × 2560</span>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {box && (
              <button
                onClick={() => setBox(null)}
                className="font-mono-ui text-[10px] uppercase tracking-[0.22em] text-cool-grey hover:text-offwhite"
              >
                CLEAR
              </button>
            )}
            <SignalButton
              disabled={!validSize}
              onClick={() => {
                if (!validSize || !box) return;
                onAnalyze({
                  widthPx: Math.round(box.w),
                  heightPx: Math.round(box.h),
                  lng: center.lng,
                  lat: center.lat,
                  zoom,
                });
              }}
            >
              {overSize
                ? "Exceeds Limit"
                : !box
                  ? "Draw Selection"
                  : "Analyze Selection →"}
            </SignalButton>
          </div>
        </div>
      </div>
    </section>
  );
}

function SelectionBox({
  box,
  over,
  drawing,
  cursor,
}: {
  box: Box;
  over: boolean;
  drawing: boolean;
  cursor: { x: number; y: number } | null;
}) {
  const color = over ? "var(--danger)" : "var(--signal)";
  return (
    <>
      {}
      <div
        className="pointer-events-none absolute"
        style={{
          left: box.x,
          top: box.y,
          width: box.w,
          height: box.h,
          boxShadow: "0 0 0 9999px rgba(11,21,36,0.55)",
          transition: drawing ? "none" : "all 120ms ease-out",
        }}
      />
      {}
      <div
        className="pointer-events-none absolute"
        style={{
          left: box.x,
          top: box.y,
          width: box.w,
          height: box.h,
          border: `1.5px solid ${color}`,
          background: over
            ? "rgba(255,59,48,0.10)"
            : "rgba(255,107,26,0.10)",
          boxShadow: over
            ? "0 0 0 1px rgba(255,59,48,0.35), 0 0 32px rgba(255,59,48,0.35)"
            : "0 0 0 1px rgba(255,107,26,0.25), 0 0 24px rgba(255,107,26,0.20)",
          animation: over
            ? "flash-warn 700ms ease-in-out infinite"
            : undefined,
        }}
      >
        {}
        {["tl", "tr", "bl", "br"].map((c) => (
          <span
            key={c}
            className="absolute h-2.5 w-2.5"
            style={{
              background: color,
              left: c.includes("l") ? -5 : undefined,
              right: c.includes("r") ? -5 : undefined,
              top: c.includes("t") ? -5 : undefined,
              bottom: c.includes("b") ? -5 : undefined,
              boxShadow: `0 0 8px ${color}`,
            }}
          />
        ))}

        {}
        <div
          className="pointer-events-none absolute -top-6 left-0 font-mono-ui text-[10px] uppercase tracking-[0.18em]"
          style={{ color }}
        >
          W {Math.round(box.w)} PX
        </div>
        <div
          className="pointer-events-none absolute -left-2 top-0 -translate-x-full origin-bottom-left font-mono-ui text-[10px] uppercase tracking-[0.18em]"
          style={{ color }}
        >
          H {Math.round(box.h)}
        </div>
      </div>

      {}
      {drawing && cursor && (
        <div
          className="pointer-events-none absolute z-10"
          style={{ left: cursor.x + 14, top: cursor.y + 14 }}
        >
          <div
            className="glass-panel rounded-sm px-2.5 py-1.5 font-mono-ui text-[11px] uppercase tracking-[0.2em]"
            style={{
              color: over ? "var(--danger)" : "var(--offwhite)",
              borderColor: over ? "var(--danger)" : undefined,
              animation: over
                ? "flash-warn 600ms ease-in-out infinite"
                : undefined,
            }}
          >
            {Math.round(box.w)} × {Math.round(box.h)} PX
            {over && (
              <div className="mt-0.5 text-[9px] tracking-[0.28em]">
                ▲ EXCEEDS 2560 LIMIT
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}

function TransitionOverlay({ box }: { box: Box }) {
  return (
    <div
      className="pointer-events-none absolute inset-0 z-30"
      style={{ animation: "vignette-in 750ms ease-out forwards" }}
    >
      {}
      <div
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(circle at center, rgba(11,21,36,0) 0%, rgba(6,16,30,0.9) 55%, #06101E 100%)",
          animation: "vignette-in 750ms ease-out both",
        }}
      />
      {}
      <div
        className="absolute"
        style={{
          left: box.x,
          top: box.y,
          width: box.w,
          height: box.h,
          border: "1.5px solid var(--signal)",
          boxShadow:
            "0 0 0 9999px rgba(6,16,30,0.75), 0 0 40px rgba(255,107,26,0.35)",
          animation: "aperture-in 800ms cubic-bezier(0.22,1,0.36,1) forwards",
        }}
      />
    </div>
  );
}