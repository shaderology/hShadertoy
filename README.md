# hShadertoy

**Experimental Shadertoy.com importer for Houdini 21 Copernicus**

https://youtu.be/ULpn8tGFsRI

## Project Overview

"You know what would be cool, if you could import a Shadertoy shader to Houdini COPs, do you think it's possible?"

1. **Editor** Shadertoy Mini IDE. input/output shadertoyAPI JSON.
2. **Builder** Sets hShadertoy HDA parameters and uses the transpiler to set OpenCL code.
3. **Transpiler** Translates GLSL to OpenCL - madness! never been done!

## Shadertoy features implementation in Houdini Digital Asset (HDA)

- GLSL fragment shaders by OpenCL nodes
- Buffer shaders by Block Begin/Block End
- Cubemaps and Volumes packed and unpacked to 2D image maps
- iMouse by hou.ViewerState or CHOPs Mouse or SOP Draw curves (todo)
- Webcam by Live Video Copernicus node (todo)
- Audio by CHOP Audio in (todo)


## Installation

1. [**Download**](https://github.com/shaderology/hShadertoy) and unpack hShadertoy to a local directory (eg: `C:/dev/hShadertoy` )

2. **Install tree-sitter:**
    > `pip install tree-sitter`

3. **Install tree-sitter-glsl:**
    > `pip install tree-sitter-glsl`

4. **Copy** `houdini/packages/hShadertoy.json` to your package directory (eg `$HOUDINI_USER_PREF_DIR/packages` )

5. **Configure** `hShadertoy.json` and set the following 3 env variables:

    ```
    { "HSHADERTOY_ROOT": "C:/dev/hShadertoy" },
    { "HSHADERTOY_HOUDINI": "C:/dev/hShadertoy/houdini" },
    { "SHADERTOY_API_KEY": "YOUR_API_KEY"},
    ```
    Get your API key: https://www.shadertoy.com/howto

    **Note:** Shadertoy API is currently blocked. Importing shaderts using API is not working.

6. Open **Editor** in hShadertoy shelf

### Known limitations
- Shadertoy global uniforms (iChannel#, iTime.. ) used outside mainImage() will be transpiled as undefined variables and produce incomplete shader
- dFdx(), dFdy() and fwidth() are just passthrough functions. (not supported in OpenCL)
- **fragColor** and **fragCoord** are expected in mainImage(). Custom names will fail. This will be fixed at some point in near future. Prob needs a simple RegEx parsing in transpile_glsl.py

## Development

### Required
```
tree-sitter>=0.25.0
tree-sitter-glsl>=0.2.0
numpy>=2.3.0
pytest>=8.4.0
pytest-cov>=7.0.0
black>=24.0.0
pylint>=3.0.0
pyopencl>=2025.2
```

### Pipeline
1. `houdini/toolbar/hShadertoy.shelf`
2. `houdini/scripts/python/hshadertoy/gui/editor.py`
3. `houdini/scripts/python/hshadertoy/builder/builder.py`
4. `houdini/scripts/python/hshadertoy/transpiler/transpile_glsl.py`
5. `src/glsl_to_opencl`
6. `houdini/otls/hShadertoy.hda` - `hShadertoy::shadertoy`
7. `magic!`

### GLSL to OpenCL specification:
- [src/glsl_to_opencl/GLSL_TO_OPENCL_SPEC.md](src/glsl_to_opencl/GLSL_TO_OPENCL_SPEC.md)

### Unit Tests
- Location: `tests/unit/`
- Coverage: All transformation features
- Run: `python -m pytest tests/unit/ -v`

### Full Transpilation Test
- `python tests/transpile.py tests/../<file>.glsl`
- outputs `<file>.header.cl` and `<file>.kernel.cl` for compilecl.py

### OpenCL Compilation Test
- Location: `tests/shaders/`
- Configure: `tests/build_options.json` 
  You can extract full list of build options by setting `HOUDINI_OCL_REPORT_BUILD_LOGS = 2` in `houdini.env`. hShadertoy HDA OpenCL node will log build options specific to your environment (more info: https://www.sidefx.com/docs/houdini/ref/env.html)
- Run: `python tests/compilecl.py --header <file>.header.cl <file>.kernel.cl`


## Known BUGS
- matrix operations not fully covered
- all matrix functions and operations are still in dev and prone to fail.
- void() functions still need polishing (foo.x needs to become foo[0] etc)


## TODO

### Transpiler:
- matrix operations attribute overload
- fix void() bugs
- investigate c++ for opencl (matrix operator overload)
- texture() functions and helper runtime update (volumes, sound etc)

### Builder
- fix video/volume parameter mapping mismatch

### HDA
- Optimize! currently evaluates the full graph even if buffer shaders are not used. Just needs simple switch.
- Pack remaining cubemaps (use included HDA hShadertoy::cubemappack)
- Pack volumes. Similar packing to cubemaps, pack z slices into 2D image grid layout.


