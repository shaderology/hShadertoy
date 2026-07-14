# Shadertoy site-source ground truth — and where hShadertoy diverges

*2026-07-08. Source of truth: the saved page source
`resources/shadertoy/wfffRN.html` (the live site's JS, WebGL2 path). All line
numbers below are into that file; they are stable because the file is a frozen
snapshot. This document exists so that transpiler/runtime decisions are made
against what the site ACTUALLY does, not against folklore. Companion:
`docs/handover/ENTRYPOINT_REDESIGN.md` (the entry-point model built from §1
here) and its §8 post-merge review.*

Audience: any model implementing the action items in §4. Each divergence
lists site behavior (with line ref), our behavior (with file ref), user-visible
impact, and a concrete prescription. None of these are compile failures —
they are render-correctness and coverage items, mostly invisible to the
compile-gated campaign.

---

## 1. How the site assembles a shader (exact, per pass type)

Every pass is compiled as ONE fragment shader built by **plain string
concatenation** — the site never parses, splits, or wraps user code:

```
renderer prefix  +  pass header  +  Common tabs (each + '\n')  +  user pass code
```

- **Renderer prefix** (line ~2834, WebGL2): `#version 300 es`, `precision
  highp float; precision highp int; precision mediump sampler3D;`. (The
  WebGL1 branch at ~2848 adds polyfills — `round`, 2-arg `trunc`,
  `transpose`, `determinant`, `inverse`, `sinh`-family, `texture()` aliases —
  these do NOT exist on the WebGL2 path; do not treat them as builtins.)
- **Pass header** (`MakeHeader_Image` line 16715, `_Buffer` 16822, `_Cubemap`
  16887, `_Sound` 16940, `_Common` 16996): uniforms, a **forward declaration**
  of the entry (`void mainImage( out vec4 c, in vec2 f );`), and the real
  `void main(void)` that CALLS it. The GLSL linker resolves the forward
  declaration to whatever definition user code provides — including one
  produced by the real GL preprocessor expanding user macros. **Declare +
  call; never splice.**
- **Assembly** (`NewShader_Image` 17189): buffers use the *Image* assembly
  path with the *Buffer* header (17285). Common tabs are compiled into every
  pass, before user code (17200-17203).

### The per-pass `main()` bodies (semantics we must match)

| Pass | site `main()` does | line |
|---|---|---|
| Image | `shadertoy_out_color = vec4(1.0); vec4 color = vec4(1e20); mainImage(color, gl_FragCoord.xy);` st_assert channel checks; **`outColor = vec4(color.xyz, 1.0)` — alpha FORCED to 1.0** | 16763-16773 |
| Buffer | `vec4 color = vec4(1e20); mainImage(color, gl_FragCoord.xy); outColor = color;` — **alpha preserved** | 16852-16859 |
| Cubemap | ray origin `ro = unCorners[4]`, direction = normalized bilinear `mix` of the 4 face-corner vectors by viewport uv; `mainCubemap(color, gl_FragCoord.xy-unViewport.xy, ro, rd); outColor = color;` (alpha preserved) | 16920-16933 |
| Sound | `t` and sample index from gl_FragCoord over a 512-wide texture; `vec2 y = mainSound(s, t);` stereo in [-1,1] packed 16-bit little-endian into RGBA (`vl`,`vh` per channel) | 16965-16989 |
| Common | compiled standalone only as a syntax check with a dummy `main` (17010-17016); its real life is textual inclusion into the other passes | 16996 |

`mainVR` is **disabled on the current site** — the VR footer is commented out
(16796-16817, 17210-17240) and `mSupportsVR=false` (17191). Excluding mainVR
from our headers is faithful; shaders defining it get it treated as a plain
helper, which matches the site.

### What the pass header guarantees user code (the "site contract")

1. Uniforms: `iResolution, iTime, iChannelTime[4], iMouse, iDate,
   iSampleRate, iChannelResolution[4], iFrame, iTimeDelta, iFrameRate`
   (16721-16730) and per-channel samplers whose TYPE follows the bound input
   (`sampler2D` / `samplerCube` / `sampler3D`, 16737-16740). Sound passes get
   a different, smaller set + `iTimeOffset`/`iSampleOffset` (16946-16951).
2. `#define HW_PERFORMANCE 0|1` — **always defined**, in every pass type
   (16719, 16826, 16891, 16944). 1 on desktop, 0 on low-end/mobile.
3. `void st_assert(bool)` and `void st_assert(bool, int)` — declared AND
   defined, **Image pass header only** (16755-16762).
4. New-API channel structs, **Image pass only** (16742-16752):
   `uniform struct { samplerXX sampler; vec3 size; float time; int loaded; } iCh0..3;`
   (see shadertoy.com/view/wtdGW8).
5. `gl_FragCoord.xy` is the GL **pixel center**: pixel (i,j) sees
   (i+0.5, j+0.5). `fragCoord` is exactly that (16767).
6. The out-param wire is `vec4 color = vec4(1e20)` (16766) — a shader that
   never writes `fragColor` shows a saturated-white blowout on the site, not
   black. (GLSL `out` params are formally undefined at function entry, but
   this is what implementations observably do.)

---

## 2. Divergence list (site vs hShadertoy)

Severity: how wrong a render can silently be. Owner column: TRANS = transpiler
hosts / package (any session may fix); HDR = `houdini/ocl/include/*.h`
(live-editable, takes effect in campaign + Houdini instantly); HDA/SCAFFOLD =
`tests/ocl/main_header.cl` generated region + HDA — **ask the owner** (see
fix-campaign skill rules).

### D1 — Image pass must force alpha to 1.0 · severity HIGH · HDA/SCAFFOLD
Site: `vec4(color.xyz, 1.0)` on Image (16772); Buffer/Cubemap preserve alpha.
Ours: `AT_fragColor_set(fragColor)` writes user alpha unchanged on every pass
(`tests/ocl/main_header.cl` SHADERTOY_INPUTS / kernel footer). 43 corpus
shaders touch `fragColor.a`; golf shaders often leave `.w` garbage — the site
hides that, we don't. In Houdini, downstream COP compositing multiplies the
damage. Prescription: on **Image passes only**, set `fragColor.w = 1.0f`
before the final write (scaffold-side, since pass type is known there;
alternatively the hosts could append it to Image kernel bodies — but the
scaffold is the single place that knows it is the Image pass).

### D2 — fragCoord must be the pixel CENTER · severity HIGH · HDA/SCAFFOLD
Site: `gl_FragCoord.xy` = (i+0.5, j+0.5) (§1.5). Ours: the unbound fallback is
`fragCoord = (float2)(AT_ix, AT_iy)` — the integer CORNER
(`tests/ocl/main_header.cl` ~1463). Everything is half a pixel off; the killer
case is buffer feedback: `texture(iChannelN, fragCoord/iResolution.xy)`
self-reads sample the texel corner → bilinear average of 4 texels → the
buffer **blurs a little more every frame** (progressive/accumulation shaders
degrade visibly). `texelFetch`/`ivec2(fragCoord)` idioms survive either way
(truncation), which hides the bug from casual checks. Prescription:
`(float2)(AT_ix + 0.5f, AT_iy + 0.5f)`; also verify what a BOUND fragCoord
layer supplies in the HDA. Render-compare's calibration gradient will
quantify this on day one.

### D3 — `#define HW_PERFORMANCE 1` · severity MED · TRANS (cheap!)
Site: always defined (§1.2). Ours: never. In C/OpenCL preprocessing an
undefined identifier in `#if` evaluates to 0, so `#if HW_PERFORMANCE==0`
**silently selects the low-quality/mobile path**, and `#ifdef HW_PERFORMANCE`
guards go dark. 0 hits in the oldest-999 corpus, but modern shaders (iq's AA
loops etc.) use it heavily — this WILL bite when the corpus moves forward.
Prescription: both hosts prepend to the emitted header:
`#ifndef HW_PERFORMANCE\n#define HW_PERFORMANCE 1\n#endif`. No HDA change,
no owner gate, near-zero risk.

### D4 — `st_assert` stubs · severity LOW · HDR
Site: two overloads, Image pass (§1.3). Ours: absent → any shader calling it
is a COMPILE_FAIL (would show as unknown-function). Prescription: no-op
overloadable stubs in `houdini/ocl/include/glslHelpers.h`:
`static void __attribute__((overloadable)) st_assert(bool c) {}` (+ `(bool,
int)`). 0 corpus hits today; costs two lines.

### D5 — out-param wire init `vec4(1e20)` vs `(0,0,0,1)` · severity LOW · HDA/SCAFFOLD
Site: §1.6. Ours: `float4 fragColor = (float4)(0,0,0,1)` (main_header.cl
~1464), and the custom-name alias in both hosts initializes the same. A shader
that reads-before-write or never writes renders differently (black vs white
blowout). Matching the site means initializing to `(float4)(1e20f)` and
accepting that Image passes then rely on D1's alpha force. Do this together
with D1 or not at all — 1e20 alpha without D1 would be catastrophic.

### D6 — `iCh0..3` struct uniforms (new API) · severity LOW · TRANS+HDR
Image passes only (§1.4). 0 corpus hits. Defer until a real shader needs it;
when it does: a small struct in the runtime header + transpiler mapping
`iChN.size`→`(float3)(AT_iChannelN_res,0)`, `.time`→`AT_Time`, `.loaded`→1.

### D7 — faithful already (no action)
- `iChannelTime[0..3] = iTime` — site semantics are per-channel, ours
  approximates with global time; acceptable until video/audio channels exist.
- mainVR exclusion (§1, disabled site-side).
- Common-tab textual inclusion before pass code — identical model.
- Per-channel sampler TYPE (cube/3D) — handled since category Z mapped
  samplers to `const IMX_Layer*`; cubemap-vs-2D dispatch lives in
  textureHelpers.h overloads.

### D8 — Sound pass reference (future)
If mainSound support is ever built, the site's packing (§1 table, 16965-16989)
is the spec: 512-wide texture, `s = iSampleOffset + y*512 + x`, output RGBA =
16-bit LE (lo,hi)×(L,R), `y` clamped [-1,1] via `floor((0.5+0.5*y)*65536)`.

---

## 3. Entry-point implications (why normalize_entry_point exists)

Because the site never scans user code (§1), any construct that RESOLVES to a
`mainImage` definition after real preprocessing is legal: `#define main()
mainImage(out vec4 fragColor, vec2 fragCoord)` + `void main(){...}`, macro
aliases of `gl_FragCoord`, etc. Our pipeline (which must find a literal
definition) approximates this with `normalize_entry_point()` in both hosts.
Its known defects and their fixes are catalogued in
`ENTRYPOINT_REDESIGN.md §8` — read that before touching it.

---

## 4. Prioritized actions

| # | Item | Owner | Effort | Gate |
|---|---|---|---|---|
| 1 | D3 HW_PERFORMANCE define | TRANS | ~30 min, both hosts + unit test | unit + corpus zero-regression |
| 2 | D4 st_assert stubs | HDR | 10 min | compile smoke |
| 3 | D2 fragCoord +0.5 | HDA/SCAFFOLD | small edit, **owner approval** | render-compare gradient + a feedback shader |
| 4 | D1 Image alpha force | HDA/SCAFFOLD | small edit, **owner approval** | render-compare on a `.a`-writing shader |
| 5 | D5 1e20 init | HDA/SCAFFOLD | with D1 only | render-compare |
| 6 | ENTRYPOINT_REDESIGN §8 F-items | TRANS | see there | unit + corpus |
| 7 | D6 iCh structs | defer | — | — |

Items 3-5 change Houdini-side render output for ALL shaders — schedule them
before render-compare baselines are frozen, or every baseline shifts.
