import hou, json, os

_SKIP_PARMS = {"generatedcode"}  # optional: skip noisy auto-generated parms
INCLUDE_EXTERNAL_CONNECTIONS = False  # set True to store edges to non-selected nodes too
SKIP_DEFAULTS = False

def run(_kwargs=None):
    sel = hou.selectedNodes()
    if not sel:
        hou.ui.displayMessage("Select one or more nodes to save.", title="No Nodes")
        return

    sel_paths = {n.path() for n in sel}

    bundle = {"version": 2, "nodes": [], "connections": []}

    # ---- collect nodes + parm data (recipe; preserves expressions/keyframes) ----
    for node in sel:
        parms = {}
        for p in node.parms():
            if p.name() in _SKIP_PARMS:
                continue
            # Only check isAtDefault() if SKIP_DEFAULTS is True
            if SKIP_DEFAULTS and p.isAtDefault():
                continue
            parms[p.name()] = p.asData(brief=True)

        bundle["nodes"].append({
            "node_name": node.name(),
            "node_type": node.type().name(),
            "node_path": node.path(),
            "parameters": parms
        })

    # ---- collect connections (only those where dst is selected; src also selected unless opted in) ----
    for dst in sel:
        for conn in dst.inputConnections():  # list of connections feeding this node
            src = conn.inputNode()           # upstream node (may be None)
            if src is None:
                continue
            if not INCLUDE_EXTERNAL_CONNECTIONS and src.path() not in sel_paths:
                continue
            # use indices straight from the connection
            in_idx  = conn.inputIndex()      # destination input index
            out_idx = conn.outputIndex()     # source output index
            bundle["connections"].append({
                "dst": dst.path(),
                "dst_input_index": in_idx,
                "src": src.path(),
                "src_output_index": out_idx
            })

    # ---- write JSON ----
    start_dir = os.path.join(hou.expandString("$HOUDINI_USER_PREF_DIR"), "scripts")
    path = hou.ui.selectFile(
        start_directory=start_dir,
        title="Save Nodes + Connections to JSON",
        pattern="*.json",
        default_value=(sel[0].name() if len(sel) == 1 else "nodes") + ".json",
        chooser_mode=hou.fileChooserMode.Write
    )
    if not path:
        return
    if not path.lower().endswith(".json"):
        path += ".json"

    with open(path, "w") as f:
        json.dump(bundle, f, indent=4)

    hou.ui.displayMessage(
        f"Saved {len(bundle['nodes'])} node(s) and {len(bundle['connections'])} connection(s) to:\n{path}",
        title="Save Complete"
    )
