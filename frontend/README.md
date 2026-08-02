# Urbaneuron Scan

Build Urbaneuron — an urban satellite segmentation platform. React + TypeScript + Tailwind. Core feature: user finds a location on a map, draws a bounding box over a city area (max 2,560×2,560 px), then the app transitions into an isolated "scanning" analysis view implying an AI segmentation model is processing the image.

Tone: precise, technical, mission-control/satellite-ops — not a generic playful SaaS template.

BRAND & COLORS — Navy dominant, Orange as signal accent (not 50/50):

Navy base: #0B1524 to #1B2A4A, used for all backgrounds/panels/chrome. Use subtle gradients (#0B1524 → #14213D), never flat fill.

Orange accent: #FF6B1A to #FF8C42 — ONLY for the scanning animation, active selection box, primary CTAs, and live/active states. It should signal "the system is doing something," never used decoratively elsewhere.

Neutrals: off-white #F4F6FA text, cool grey #5A6472 secondary text, desaturated navy-grey #2A3652 borders. No pure black/white.

Dark mode only — do not build a light mode toggle.

TYPOGRAPHY: Headings in a geometric/grotesk sans (Space Grotesk or Inter Tight). Body/UI in Inter. Coordinates, pixel dimensions, and data readouts in monospace (JetBrains Mono) to reinforce precision.

CORE USER FLOW (build this as the centerpiece):

Hero: full-bleed satellite map background, navy-tinted/desaturated, bold headline, one orange CTA that leads into the map tool.

Map + search: interactive map (Mapbox GL or MapLibre) in a custom dark navy style, not default map colors. Floating glass/blur "console" search bar accepts place names or raw lat/lng coordinates, orange focus ring on input. Custom-styled zoom/pan controls matching the navy UI.

Area selection: click-drag to draw a bounding box. Hard cap 2,560×2,560 px. Show a live monospace pixel-dimension readout following the cursor while dragging. If the user exceeds the limit, give immediate visible feedback (box edges glow red-orange, dimension label flashes) instead of silently failing. Style the box with a thin orange outline, 8–12% orange fill, small orange square corner handles for resizing. A prominent orange "Analyze Selection" button appears once size is valid.

Transition animation: on confirm, the rest of the map fades/blurs/darkens (navy vignette closing inward) while the selected area scales into a centered isolated frame, like a camera aperture closing. ~600–900ms eased. Map UI fades out simultaneously.

Scanning animation: the isolated image sits centered in a thin navy-orange gradient border. An animated scanline (glowing orange bar, 2–4px, soft trailing glow) sweeps top-to-bottom or oscillates on loop, ~2–3s per pass. Add a faint pulsing grid overlay, orange corner focus-brackets (like a camera UI), and a monospace status line below (e.g. "ANALYZING SEGMENT 3/8..."). This should feel like active technical processing, not a generic spinner.

Result view: show the segmented output with a color-coded mask overlay (buildings=orange, vegetation=muted teal, water=soft blue, roads=grey), still within the navy/orange brand. Include a before/after toggle or slider.

GENERAL PRINCIPLES:

Generous negative space; the map/image stays visually primary, UI elements feel like floating overlay consoles (glassmorphism), not stacked webpage sections.

Fast micro-interactions (150–200ms) on hovers, focus states, selection handles.

Fully responsive — coordinate search and map stack cleanly on mobile, touch-drag selection uses the same pixel-clamp feedback.

Maintain WCAG AA contrast for orange text/UI on navy backgrounds.

Keep the scanline animation transform/opacity-based for smooth performance on all devices.

For anything not explicitly specified, default toward "professional-grade satellite analysis software" over "generic dashboard template" — precise, confident, quietly high-tech.

This project was built with [Lovable](https://lovable.dev).

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/dba42ca5-dbdf-4bde-b363-d9f3024f5e6f).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
