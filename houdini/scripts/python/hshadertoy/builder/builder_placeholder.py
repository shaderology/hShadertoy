import hou, json
import hshadertoy.transpiler.transpiler_placeholder as transpiler_placeholder
from hshadertoy.builder.builder import _safe_node_name

def builder(payload, mode="Template"):

    shader_json = json.loads(payload)
    shader_name = shader_json["Shader"]["info"]["name"]
    code_glsl = shader_json["Shader"]["renderpass"][0]["code"]

    code_transpiled = transpiler_placeholder.transpile(code_glsl, mode=mode)

    # Ensure /obj exists
    obj = hou.node("/obj")
    # Create a COP network under /obj
    copnet = obj.createNode("copnet", _safe_node_name(shader_name) )
    # Create the hShadertoy node inside the COP network
    node = copnet.createNode("hShadertoy::shadertoy", "shadertoy1" )
    print(f"hShadertoy HDA node created: {node.path()}\n")

    # Set the parameters
    code_parm_name = "code_rp0"
    node.parm(code_parm_name).set(code_transpiled)
    print(f"Parameter '{code_parm_name}' set with value:\n {code_transpiled}\n")    

    return node
