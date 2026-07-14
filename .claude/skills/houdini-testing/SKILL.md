---
name: houdini-testing
description: Test the Houdini side of hShadertoy headlessly (hython, no GUI) — builder test, real OpenCL cook test, the live-header vs HDA-owner boundary, main_header.cl anatomy, and a Houdini glossary for engineers who have never opened Houdini. Use when validating HDA builds, cubemap/mainCubemap paths, runtime-header changes, cook failures, or anything involving hython, COPs, #bind, or IMX_Layer.
---

# Headless Houdini testing

The HDA (`houdini/otls/hShadertoy.hda`, binary) is **owner-maintained — never
edit it**. Everything else here is testable by you, without the Houdini GUI.

> **Never used Houdini?** Read the Glossary at the bottom of this file FIRST —
> the rest of this skill uses COP/HDA/`#bind`/IMX_Layer/VEX freely.

## The test levels (cheapest first)

### Level 0 — no Houdini at all: `tests/compilecl.py` (pyopencl)
```bash
python tests/compilecl.py --header <x.header.cl> <x.kernel.cl>
```
Defaults: `tests/ocl/main_header.cl`, `tests/ocl/main_kernel.cl`,
`tests/build_options.json`. Emulates the COP compile environment (source order:
main_header → your header → main_kernel → your kernel → closing
`AT_fragColor_set(fragColor);}`). Needs pyopencl + a GPU driver, no license.
**Gotcha:** a shader's Common pass must be merged into the transpiled header
(prepended to the GLSL *before* transpiling — the campaign does this); otherwise
you get implicit-declaration warnings + ptxas unresolved externs.

### Level 1 — HDA build + transpile, NO cook: `builder_test_headless.py`
```bash
HSHADERTOY_ROOT="C:/dev/hShadertoy" \
"/c/Program Files/Side Effects Software/Houdini 21.0.440/bin/hython.exe" \
  houdini/scripts/python/hshadertoy/tests/builder_test_headless.py \
  "C:/dev/hShadertoy/resources/examples/BuffersAndTextures/BuffersAndTextures_API.json" Transpile
```
- Houdini is **21.0.440** (not 20.5). `HSHADERTOY_ROOT` drives sys.path AND the
  explicit `hou.hda.installFile` — the package env is not auto-applied headless.
- Arg 1 must be **API-shaped** JSON: `{"Shader": {...}}`. The campaign cache
  files (`tests/campaign/cache/<id>.json`) are the INNER dict — wrap them first.
- Arg 2: `Transpile` (production transpiler) or `Template` (placeholder).
- Validates: JSON parsing, renderpass tokenization, transpile of every pass,
  HDA parm population. Does **NOT** compile OpenCL (the cook is commented out
  in `builder.py`). Success prints `SUCCESS!`; failure = traceback, exit 1.

### Level 2 — HDA build + real cook: `builder_cook_headless.py` (PREFERRED)
```bash
python tests/fixcampaign/houdini_smoke.py    # one-command wrapper, exit code is the verdict
# or directly:
HSHADERTOY_ROOT="C:/dev/hShadertoy" HOUDINI_OCL_PATH="C:/dev/hShadertoy/houdini/ocl;&" \
hython houdini/scripts/python/hshadertoy/tests/builder_cook_headless.py \
  "C:/dev/hShadertoy/resources/examples/BuffersAndTextures/BuffersAndTextures_API.json" Transpile
```
- Same API-shaped input as Level 1, but after building it runs
  `hou.node().cook(force=True)` on the HDA node — Houdini compiles AND executes
  the OpenCL of every enabled renderpass (incl. cubemap/Common paths the
  campaign never sees). Collects `errors()` from the HDA + all subchildren and
  exits non-zero on any (verified both directions 2026-07-05).
- **CRITICAL — `HOUDINI_OCL_PATH` must end in `;&`.** The `&` expands to
  Houdini's default OCL search path. Setting the var WITHOUT it removes the
  built-in headers (`imx.h`, `interpolate.h`, …) and the APEX-implemented COPs
  inside the HDA then **SEGFAULT hython** (UT_Lock in `CE_SnippetCacheEntry::
  lookupCode`) instead of erroring. Every `#import`-cannot-be-found or
  seg-fault-on-cook headless: check this first.
- Run via Bash `run_in_background` with timeout 600000; never pipe through
  `tail` (the pipe eats the exit code).
- Wired into the fix campaign as the mandatory post-corpus step
  (`tests/fixcampaign/README.md` step 8).

### Level 2b — template-shaped cook: `template_load_headless.py`
```bash
hython houdini/scripts/python/hshadertoy/tests/template_load_headless.py \
  --cook opencl "resources/examples/BuffersAndTextures/BuffersAndTextures_HDA.json"
```
- Input is an **HDA-template** JSON (`{"nodes":[...], "connections":[...]}`),
  exported via `houdini/scripts/python/hshadertoy/tools/template_save.py` —
  NOT the API shape.
- `--cook opencl` force-cooks the OpenCL COPs = Houdini actually compiles and
  runs the kernels. The cook needs `HOUDINI_OCL_PATH=<repo>/houdini/ocl;&` set
  (note the `;&` — see Level 2 warning) to resolve the `#include`s.
- NOTE: the template JSON carries *frozen pre-transpiled* code in its parms —
  it does NOT exercise the current transpiler. For transpiler validation use
  Level 2 above.
- **KNOWN BUG (roadmap quick-win):** the script collects cook errors but never
  prints them and always exits 0 — its "success" summary prints *before*
  cooking. Until fixed, check hython stderr and query `node.errors()` yourself;
  do not trust a clean-looking run.
- There is currently **no image-output/pixel-diff harness** — render
  correctness today is "cook succeeds + a human looks at it". Building that is
  the render-compare skill's Phase 1.

## Canonical full-stack shader: `wfffRN` (BuffersAndTextures)

Written specifically for hShadertoy: image + Buffer A/B/C + Cube A + Common in
one shader. Local copies (no API needed):
`resources/examples/BuffersAndTextures/BuffersAndTextures_API.json` (already
`{"Shader":...}`-wrapped) and `..._HDA.json` (template shape). **Use it to
validate any transpiler change that could affect non-mainImage entry points** —
the mass-test campaign is mainImage-only, so `mainCubemap`/`mainSound`
regressions are invisible there (this happened once: a dangling
`__attribute__((overloadable))` broke the cubemap pass while the campaign
stayed green).

## The live-vs-owner boundary (get this right)

**Live-editable — edits take effect immediately in BOTH the campaign compile
and live Houdini renders, no HDA regeneration:** the four real runtime headers
in `houdini/ocl/include/`: `glslHelpers.h`, `textureHelpers.h`,
`matrix_types.h`, `matrix_ops.h`. Campaign resolves them via
`tests/build_options.json` `-I`; Houdini via package env `HOUDINI_OCL_PATH`.
Nothing is flattened or embedded.

**Owner-must-regenerate (HDA `code_header` structure):** `#bind` lines, the
`static` globals with VEX backtick initializers, `SHADERTOY_INPUTS`,
`CUBEMAP_RENDERPASS`/`DO_CUBEMAP`, `shadertoy_cubemap()`. These live inside the
binary HDA. `houdini/ocl/include/shadertoyInputs.h` is a **read-only
documentation MIRROR of that code_header — nothing #includes it; editing it
changes nothing at runtime.** (Older text in `tests/fixcampaign/README.md` and
`HOUDINI_HANDOFF.md` listing it among live headers is wrong.) For structural
changes: write the exact diff + steps into `tests/fixcampaign/HOUDINI_HANDOFF.md`,
mark the backlog item `BLOCKED-ON-HOUDINI`, and ask the owner.

**`tests/ocl/main_header.cl` anatomy:** lines ~1–1401 are Houdini-generated
`#bind`-expansion plumbing (`AT_*`/`_bound_*` macros) captured from a real COP;
the `HSHADERTOY BEGIN` marker (~line 1404) starts the expanded code_header with
VEX resolved to constants (`iResolution=(512,288)`, `iTime=0`). If the owner
changes the HDA code_header, this snapshot AND `shadertoyInputs.h` must be
re-captured or the campaign drifts from the real runtime. Never hand-edit the
generated region — ask the owner.

## Environment notes

- Package file `houdini/packages/hShadertoy.json`: the checked-in copy has
  empty env values and `"enable": false` — it's a template. A live install
  copies it to `$HOUDINI_USER_PREF_DIR/packages` with real `HSHADERTOY_ROOT`
  and enables it. Headless Level-1 works without it; Level-2 cooking needs
  `HOUDINI_OCL_PATH`.
- hython checks out a Houdini license at startup — if none is free the script
  dies immediately. Startup costs tens of seconds; budget Bash timeouts
  accordingly.
- Transpiling inside hython needs `tree-sitter` + `tree-sitter-glsl` importable
  by Houdini's Python 3.11 (install with
  `<houdini>/python311/python.exe -m pip install --user <pkg>`).

## Glossary (for engineers who've never opened Houdini)

- **COP / Copernicus** — Houdini's GPU image-compositing node context; each
  Shadertoy renderpass becomes an OpenCL COP whose kernel compiles at **cook**
  (evaluation) time, not node-creation time.
- **HDA** — Houdini Digital Asset: the packaged `hShadertoy::shadertoy` node
  type. Binary file; owner-only.
- **`code_header`** — a string parameter (UI field) on the HDA's OpenCL COP
  nodes holding the OpenCL source prepended before the user kernel: the
  `#bind` lines, static globals, `SHADERTOY_INPUTS` macro. It lives inside the
  binary HDA; `shadertoyInputs.h` is its read-only mirror.
- **VEX / backtick expressions** — Houdini's expression language. Backticked
  expressions inside a string parm (e.g. `` `rint(ch('init_iResolutionx'))` ``)
  are evaluated by Houdini BEFORE the OpenCL compiler ever sees the text —
  that's how the static globals get "initializers" without OpenCL uniforms.
- **`hou` / node paths** — `hou` is Houdini's Python API module (only inside
  hython). Nodes live at filesystem-like paths: the headless scripts build
  `/obj/<subnet>/copnet1/...`; `hou.node(path)` fetches one, `.errors()`
  reads its cook errors.
- **`#bind`** — Copernicus pragma declaring a kernel binding to a parm
  (`#bind parm iMouse float4`) or image layer (`#bind layer iChannel0? opt
  val=0`; `?` = optional). Houdini expands each into the `HAS_x`/`AT_x_*` macro
  blocks filling main_header.cl's first 1400 lines.
- **`@x` vs `AT_x`** — HDA-side source writes `@iChannel0.layer`; Houdini
  rewrites `@` → `AT_` macros. main_header.cl shows the post-rewrite form.
- **`IMX_Layer`** — SideFX image-buffer struct (+ `stat` metadata: resolution,
  border, storage, `typeinfo`). Every GLSL `sampler2D/samplerCube` becomes
  `const IMX_Layer*`; all `texture*()` helpers in `textureHelpers.h` take it.
- **`SHADERTOY_INPUTS`** — macro that must open every kernel body: copies bound
  parms/layers into the static globals, declares `fragCoord`/`fragColor`, runs
  `DO_CUBEMAP`.
- **Static-globals trick** — OpenCL has no uniforms, so `iTime` etc. are
  `static` globals initialized with owner-side VEX expressions and re-assigned
  per pixel. Consequence: `iTime` referenced in a *global initializer* is a
  frozen constant.
- **Cubemap 3×2 packing** — `mainCubemap` passes render 6 faces into one 3×2
  image ([+X][−X][+Z] / [+Y][−Y][−Z]); `samplerCube` reads the same layout.
  Mechanics: when the HDA marks a pass as cubemap it defines
  `CUBEMAP_RENDERPASS`, making the `DO_CUBEMAP` macro (run inside
  `SHADERTOY_INPUTS`) call `shadertoy_cubemap()` to compute the per-pixel
  `rayDir` and rewrite `iResolution` to face size. Only exercised by wfffRN —
  the campaign never defines `CUBEMAP_RENDERPASS`.
- **Media** — Shadertoy textures are bundled at `houdini/pic/media/a/<hash>.png`
  (cubemap faces `_1.._5`), mapped by `houdini/pic/media.json`; do NOT try to
  download from shadertoy.com/media (see shadertoy-api skill).
- **Derivatives are fake** — `GLSL_dFdx/dFdy/fwidth` in `glslHelpers.h` are
  passthrough dummies (no derivatives in OpenCL COPs); expect visual
  differences in shaders that rely on them.
