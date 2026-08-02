import { useState, useCallback, useRef } from "react";
import { HeroView } from "./HeroView";
import { MapView } from "./MapView";
import { ScanView } from "./ScanView";
import { ResultView } from "./ResultView";
import { segmentTile, type SegmentResult } from "@/lib/inference.functions";

export type Phase = "hero" | "map" | "transition" | "scan" | "result";

export type Selection = {
  widthPx: number;
  heightPx: number;
  lng: number;
  lat: number;
  zoom: number;
};

const MIN_SCAN_MS = 4200;

export function Urbaneuron() {
  const [phase, setPhase] = useState<Phase>("hero");
  const [selection, setSelection] = useState<Selection | null>(null);
  const [result, setResult] = useState<SegmentResult | null>(null);
  const timers = useRef<number[]>([]);

  const goToMap = useCallback(() => setPhase("map"), []);

  const onAnalyze = useCallback(
    (sel: Selection) => {
      setSelection(sel);
      setResult(null);
      setPhase("transition");
      timers.current.forEach(window.clearTimeout);
      timers.current = [];

      timers.current.push(window.setTimeout(() => setPhase("scan"), 850));

      const started = Date.now();
      segmentTile(sel)
        .then((res) => res)
        .catch(
          (): SegmentResult => ({
            status: "simulated",
            note: "INFERENCE CALL FAILED",
          }),
        )
        .then((res) => {
          const elapsed = Date.now() - started;
          const wait = Math.max(0, 850 + MIN_SCAN_MS - elapsed);
          timers.current.push(
            window.setTimeout(() => {
              setResult(res);
              setPhase("result");
            }, wait),
          );
        });
    },
    [],
  );

  const reset = useCallback(() => {
    timers.current.forEach(window.clearTimeout);
    timers.current = [];
    setSelection(null);
    setResult(null);
    setPhase("map");
  }, []);

  const goHome = useCallback(() => {
    timers.current.forEach(window.clearTimeout);
    timers.current = [];
    setSelection(null);
    setResult(null);
    setPhase("hero");
  }, []);

  return (
    <main className="relative min-h-screen w-full overflow-hidden text-offwhite">
      {phase === "hero" && <HeroView onStart={goToMap} />}
      {(phase === "map" || phase === "transition") && (
        <MapView
          transitioning={phase === "transition"}
          onAnalyze={onAnalyze}
          onBack={() => setPhase("hero")}
        />
      )}
      {phase === "scan" && selection && <ScanView selection={selection} />}
      {phase === "result" && selection && (
        <ResultView
          selection={selection}
          result={result}
          onReset={reset}
          onHome={goHome}
        />
      )}
    </main>
  );
}
