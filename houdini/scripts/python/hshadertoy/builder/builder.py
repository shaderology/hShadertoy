"""
Shadertoy to Houdini HDA Builder

Translates Shadertoy API JSON to hShadertoy::shadertoy HDA parameters.
Because we're all tired of manually setting 200+ parameters.
"""

import json
import re
import hou
from typing import Dict, Any, Optional, Tuple


def _safe_node_name(name: str, fallback: str = "shadertoy") -> str:
    """
    Sanitize an arbitrary shader title into a valid Houdini node name.

    Houdini node names only allow [A-Za-z0-9_] and must not start with a
    digit. Shadertoy titles allow anything ("There's a bug in the TV ",
    "2.4GHz", "[SH17C] Schooling"...), so every run of other characters
    collapses to a single underscore.

    >>> _safe_node_name("There's a bug in the TV ")
    'There_s_a_bug_in_the_TV'
    >>> _safe_node_name("2.4GHz")
    '_2_4GHz'
    """
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    if not safe:
        safe = fallback
    if safe[0].isdigit():
        safe = "_" + safe
    return safe


# Renderpass name to HDA parameter token mapping
# Why these specific numbers? Because Shadertoy said so, that's why.
RENDERPASS_TOKENS = {
    "Image": 0,
    "Buffer A": 1,
    "Buffer B": 2,
    "Buffer C": 3,
    "Buffer D": 4,
    "Cube A": 5,
    "Sound": 6,
    "Common": "common"
}


def _load_assets_mapping() -> Dict[int, Dict[str, Any]]:
    """
    Load the asset ID to HDA parameter mapping.

    This JSON file is basically our Rosetta Stone for media files.
    Without it, we'd be mapping asset IDs manually like cavemen.

    Returns:
        Dict mapping asset IDs to their HDA folder/asset tokens
    """
    import os
    # Because hardcoded paths are bad, but environment variables are good
    assets_json_path = os.path.join(
        os.path.dirname(__file__),
        "hda",
        "assets.json"
    )

    with open(assets_json_path, 'r') as f:
        data = json.load(f)

    # Convert list to dict keyed by ID for O(1) lookup instead of O(n)
    # Your future self will thank you when processing 4 channels × 6 renderpasses
    return {item["id"]: item for item in data["inputs"]}


def _resolve_renderpass_name(renderpass: Dict[str, Any]) -> str:
    """
    Canonical renderpass name, deriving it from `type` when `name` is empty.

    Old (pre-~2016) Shadertoy API shaders return name="" (e.g. 4l2XWw
    "phyllotaxis 2D"); skipping those passes silently built a DEFAULT-state
    HDA. Type is unambiguous for image/common/cubemap/sound. A nameless
    BUFFER pass stays "" (A-D cannot be told apart from type alone; never
    seen in the 999-shader corpus) - the caller must warn, not silently skip.
    """
    name = renderpass.get("name")
    if name:
        return name
    return {
        "image": "Image",
        "common": "Common",
        "cubemap": "Cube A",
        "sound": "Sound",
    }.get(renderpass.get("type", ""), "")


def _get_renderpass_token(renderpass_name: str) -> int | str:
    """
    Convert renderpass name to token.

    Args:
        renderpass_name: The human-readable name like "Buffer A"

    Returns:
        The token (0-6 or "common")

    Raises:
        ValueError: If someone invented a new renderpass type without telling us
    """
    token = RENDERPASS_TOKENS.get(renderpass_name)
    if token is None:
        raise ValueError(
            f"Unknown renderpass '{renderpass_name}'. "
            f"Valid options: {list(RENDERPASS_TOKENS.keys())}"
        )
    return token


def _build_renderpass_params(
    renderpass: Dict[str, Any],
    rp_token: int | str,
    assets_map: Dict[int, Dict[str, Any]],
    transpiler_func: callable,
    transpiler_mode: str
) -> Dict[str, Any]:
    """
    Build HDA parameters for a single renderpass.

    This is where the magic happens. Or the suffering. Depends on your perspective.

    Args:
        renderpass: The renderpass data from Shadertoy API
        rp_token: The renderpass token (0-6 or "common")
        assets_map: The asset ID to HDA mapping
        transpiler_func: Function to transpile GLSL to OpenCL
        transpiler_mode: Mode for transpiler ("Template", "Transpiler", etc.)

    Returns:
        Dict of HDA parameter names to values
    """
    params = {}

    # Enable the renderpass (except Image which is always enabled)
    # Image is special. Image doesn't need to prove itself.
    # Common uses different parameter naming (no "rp" prefix)
    if rp_token != 0:
        if rp_token == "common":
            params[f"enable_{rp_token}"] = True
        else:
            params[f"enable_rp{rp_token}"] = True

    # Transpile and set the code
    # This is where GLSL becomes OpenCL. Cross your fingers.
    code_glsl = renderpass.get("code", "")
    if code_glsl:
        try:
            # Note: transpiler_mode is builder mode ("Template"/"Transpile")
            # For Template mode, pass it through (placeholder uses it)
            # For production transpiler, pass None to enable auto-detection of renderpass type
            if transpiler_mode == "Template":
                code_opencl = transpiler_func(code_glsl, mode=transpiler_mode)
            else:
                # Production transpiler: let it auto-detect renderpass type (mainImage, Common, etc.)
                code_opencl = transpiler_func(code_glsl, mode=None)
        except Exception as e:
            # Provide context about which renderpass failed
            rp_name = renderpass.get("name", "Unknown")
            print(f"ERROR: Failed to transpile renderpass '{rp_name}': {e}")
            raise RuntimeError(f"Transpilation failed for renderpass '{rp_name}': {e}") from e

        # Common uses "code_common", others use "code_rp{N}"
        if rp_token == "common":
            params[f"code_{rp_token}"] = code_opencl
        else:
            params[f"code_rp{rp_token}"] = code_opencl

    # Process input channels (0-3)
    # Each renderpass has 4 channels because Shadertoy is generous like that
    inputs = renderpass.get("inputs", [])
    for input_data in inputs:
        channel = input_data.get("channel")
        if channel is None:
            continue

        asset_id = input_data.get("id")
        if asset_id is None:
            continue

        # Look up the asset in our mapping
        # If it's not there, someone forgot to update assets.json. Shame on them.
        asset_info = assets_map.get(asset_id)
        if not asset_info:
            print(f"Warning: Asset ID {asset_id} not found in assets.json. Skipping.")
            continue

        folder_token = asset_info["hda"]["folder"]["token"]
        asset_token = asset_info["hda"]["asset"]["token"]

        # Set folder and asset parameters
        # Common uses different naming (no "rp" prefix)
        if rp_token == "common":
            prefix = f"{rp_token}_ch{channel}"
        else:
            prefix = f"rp{rp_token}_ch{channel}"
        params[f"folder_{prefix}"] = folder_token
        # Yes, the folder token goes in the parameter name. Don't ask why.
        params[f"asset{folder_token}_{prefix}"] = asset_token

        # Forward the Shadertoy sampler settings to the HDA channel parms.
        # Missing keys fall back to the site defaults (mipmap/repeat/vflip on).
        # HDA menu tokens: wrap Clamp='1' Wrap='3'; filter 0/1/2 = nearest/
        # linear/mipmap; the vflip toggle already matches site semantics
        # (the iChannel HDA inverts internally for Houdini's row order).
        sampler = input_data.get("sampler", {})
        params[f"vflip_{prefix}"] = \
            str(sampler.get("vflip", "true")).lower() != "false"
        params[f"wrap_{prefix}"] = \
            "1" if sampler.get("wrap") == "clamp" else "3"
        params[f"filter_{prefix}"] = \
            {"nearest": 0, "linear": 1}.get(sampler.get("filter"), 2)

    return params


def build_shadertoy_hda(
    shader_json: str | Dict[str, Any],
    transpiler_func: callable = None,
    mode: str = "Template",
    parent_node: Optional[hou.Node] = None,
    node_name: Optional[str] = None
) -> hou.Node:
    """
    Build a complete hShadertoy HDA node from Shadertoy API JSON.

    This is the main entry point. It does all the heavy lifting so you don't have to.

    IMPORTANT: The hShadertoy HDA requires COP (Copernicus) context. If no COP network
    parent is provided (or if the parent is not a COP network), this function will
    automatically create a dedicated COP network under /obj named after the shader.
    This follows the Houdini convention where example files load into their own networks.

    Args:
        shader_json: Either a JSON string or dict from Shadertoy API
        transpiler_func: Function to transpile GLSL to OpenCL. If None, selects based on mode.
        mode: Transpiler mode - "Template" (placeholder, comments out code) or
              "Transpile" (production transpiler, full GLSL->OpenCL conversion). Default: "Template"
        parent_node: Parent node (COP network). If None or not COP context, creates copnet under /obj.
        node_name: Name for the HDA node. If None, uses "shadertoy1"

    Returns:
        The created hShadertoy::shadertoy HDA node

    Example:
        >>> with open("gameOfLife_API.json") as f:
        ...     shader_data = f.read()
        >>> node = build_shadertoy_hda(shader_data, mode="Template")
        >>> print(node.path())
        /obj/Game_Of_Life/shadertoy1
    """
    # Parse JSON if it's a string
    # Because sometimes it's a string, sometimes it's a dict. Life is uncertain.
    if isinstance(shader_json, str):
        shader_data = json.loads(shader_json)
    else:
        shader_data = shader_json

    # Load assets mapping once
    # This is expensive, so we only do it once. We're not barbarians.
    assets_map = _load_assets_mapping()

    # Import transpiler based on mode if not provided
    # Template mode: Placeholder (comments out code) - useful for debugging
    # Transpile mode: Production transpiler (GLSL -> OpenCL) - for real shaders
    if transpiler_func is None:
        if mode == "Template":
            from hshadertoy.transpiler.transpiler_placeholder import transpile
            transpiler_func = transpile
        else:  # mode == "Transpile" or other values default to production
            from hshadertoy.transpiler.transpile_glsl import transpile
            transpiler_func = transpile

    # Extract shader info
    shader_info = shader_data["Shader"]["info"]
    shader_name = shader_info.get("name", "shadertoy")
    renderpasses = shader_data["Shader"]["renderpass"]

    # Ensure we have a COP network to work in
    # The HDA only works in COP context, not in /obj or anywhere else
    # This is the Houdini way - examples are loaded into their own networks
    if parent_node is None or parent_node.type().name() != "copnet":
        # Either no parent specified, or parent is not a COP network
        # Create a dedicated COP network under /obj for this shader
        obj = hou.node("/obj")
        parent_node = obj.createNode("copnet", _safe_node_name(shader_name))
        print(f"Created COP network: {parent_node.path()}")

    # Create the HDA node
    # This is the moment of truth
    hda_node_name = node_name or "shadertoy1"

    # Verify HDA type exists before attempting to create
    # Because cryptic "Invalid node type name" errors are the worst
    node_type_name = "hShadertoy::shadertoy"
    parent_type = parent_node.childTypeCategory()

    if parent_type.nodeType(node_type_name) is None:
        # Try to be helpful about what went wrong
        available_types = [nt.name() for nt in parent_type.nodeTypes().values()
                          if "shadertoy" in nt.name().lower()]

        error_msg = (
            f"HDA node type '{node_type_name}' not found in Houdini.\n\n"
            f"This usually means:\n"
            f"1. The HDA file is not installed\n"
            f"2. The HDA file is not in HOUDINI_OTLSCAN_PATH\n"
            f"3. The HDA namespace/name doesn't match\n\n"
            f"Expected: {node_type_name}\n"
            f"Context: {parent_type.name()}\n"
        )

        if available_types:
            error_msg += f"\nFound similar types: {', '.join(available_types)}\n"

        error_msg += (
            f"\nTo fix:\n"
            f"- Check that hShadertoy.hda is in: {hou.getenv('HOUDINI_OTLSCAN_PATH', 'NOT SET')}\n"
            f"- Verify the HDA contains a node named 'shadertoy' in namespace 'hShadertoy'\n"
            f"- Try: Windows > Operator Type Manager to see installed HDAs"
        )

        raise RuntimeError(error_msg)

    node = parent_node.createNode(node_type_name, hda_node_name)



    print(f"Created hShadertoy HDA node: {node.path()}")

    # Build all parameters
    # Here's where we set approximately a billion parameters
    all_params = {}

    for renderpass in renderpasses:
        rp_name = _resolve_renderpass_name(renderpass)
        if not rp_name:
            print(f"Warning: skipping renderpass with no name and "
                  f"unresolvable type {renderpass.get('type')!r}")
            continue

        try:
            rp_token = _get_renderpass_token(rp_name)
        except ValueError as e:
            print(f"Warning: {e}")
            continue

        # Build params for this renderpass
        rp_params = _build_renderpass_params(
            renderpass,
            rp_token,
            assets_map,
            transpiler_func,
            mode
        )

        all_params.update(rp_params)

    # Set all parameters on the node
    # This is where the rubber meets the road
    for parm_name, parm_value in all_params.items():
        parm = node.parm(parm_name)
        if parm is None:
            print(f"Warning: Parameter '{parm_name}' not found on HDA. HDA definition may be outdated.")
            continue

        try:
            parm.set(parm_value)
            # Uncomment for verbose debugging. Or don't. I'm not your mother.
            # print(f"Set {parm_name} = {parm_value}")
        except Exception as e:
            print(f"Error setting parameter '{parm_name}': {e}")

    print(f"Configured {len(all_params)} parameters on {node.path()}")

    # Position the node nicely
    # Because messy networks give me anxiety
    node.moveToGoodPosition()

    # Note: OpenCL compilation happens when node is cooked (rendered)
    # In headless mode, include paths may not be set up correctly
    # Full compilation testing should be done in GUI (Phase 3 Task 5)
    # node.cook(force=True)  # Commented out - compile in GUI instead

    return node


# Backward compatibility with the old placeholder API
# Because breaking existing code is mean
def builder(payload: str, mode: str = "Template") -> hou.Node:
    """
    Legacy builder function for backward compatibility.

    This exists so old code doesn't break. You're welcome.
    Please use build_shadertoy_hda() for new code.

    Args:
        payload: JSON string from Shadertoy API
        mode: Transpiler mode

    Returns:
        The created HDA node
    """
    return build_shadertoy_hda(payload, mode=mode)
