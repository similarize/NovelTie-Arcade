# froggies-render-farm (render-only branch)
Orphan branch used as a free Blender Cycles render farm for an animated short. It contains only render
scripts, shot data and the workflow; it shares no history with this repository's game code.
Push a new `job.json` and the `froggies-render` workflow splits the listed shot frames across matrix runners
(official Blender 4.3.2 x64, Poly Haven CC0 HDRIs) and uploads PNG frames as short-lived artifacts.
