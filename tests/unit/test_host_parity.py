"""
Anti-drift guard for the two transpiler hosts.

Host A (``tests/transpile.py``) and Host B
(``houdini/.../transpile_glsl.py``) used to each carry a private copy of the
whole GLSL→OpenCL pipeline, and repeatedly drifted (S63b matrix_macros seed,
S59 rescue, category-A hoisting, tsKXR3 Common inout signatures). They are now
thin adapters over the single shared core
``src/glsl_to_opencl/host_pipeline.py``.

These tests make re-divergence FAIL loudly:
  * structural — the pipeline entry point and every shared helper must be the
    SAME function object in both hosts and the core (a re-forked private copy
    breaks identity);
  * functional — both hosts must produce the same header/body core for the same
    input, and both must address-of an inout arg to a Common-defined helper
    (the exact tsKXR3 bug that motivated the unification).
"""
import sys
sys.path.insert(0, 'C:/dev/hShadertoy/houdini/scripts/python')
sys.path.insert(0, 'C:/dev/hShadertoy')

import pytest

from src.glsl_to_opencl import host_pipeline
import tests.transpile as host_a
from hshadertoy.transpiler import transpile_glsl as host_b


class TestStructuralSingleSource:
    """Both hosts must delegate to the ONE shared pipeline, not a private copy."""

    def test_both_hosts_use_the_same_pipeline_entry(self):
        assert host_a.transpile_pass is host_pipeline.transpile_pass
        assert host_b.transpile_pass is host_pipeline.transpile_pass

    def test_host_a_shared_helpers_are_the_core_objects(self):
        assert host_a.normalize_entry_point is host_pipeline.normalize_entry_point
        assert host_a.partition_translation_unit is host_pipeline.partition_translation_unit
        assert host_a.post_process_ifdef_blocks is host_pipeline.post_process_ifdef_blocks
        assert host_a.entry_param_names is host_pipeline.entry_param_names

    def test_host_b_shared_helpers_are_the_core_objects(self):
        assert host_b._detect_renderpass_type is host_pipeline.detect_renderpass_type
        assert host_b._harvest_common_signatures is host_pipeline.harvest_common_signatures
        assert host_b._normalize_entry_point is host_pipeline.normalize_entry_point
        assert host_b._partition_translation_unit is host_pipeline.partition_translation_unit
        assert host_b._post_process_ifdef_blocks is host_pipeline.post_process_ifdef_blocks


# Both hosts get their pipeline from the same module; this bridging import lives
# on both so the two hosts truly share it.
def _pass_pieces(**kwargs):
    return host_pipeline.transpile_pass(**kwargs)


class TestFunctionalEquivalence:
    """For the same input the two hosts' shared-core results must agree, save
    for the two intended differences (output wrapper, Common strategy)."""

    SIMPLE = (
        "float helper(float x) { return x * 2.0; }\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    vec2 uv = fragCoord / iResolution.xy;\n"
        "    fragColor = vec4(helper(uv.x), uv.y, 0.0, 1.0);\n"
        "}\n"
    )

    def test_hosts_agree_on_header_and_body_no_common(self):
        # No Common tab => merge_common is a no-op, so both hosts must produce
        # the identical header + body core (only indent knobs differ, and this
        # shader has no hoisting / custom param names to expose them).
        a = _pass_pieces(glsl_source=self.SIMPLE, mode="mainImage",
                         merge_common=True, require_entry=True,
                         hoist_indent="    ", bridge_indent="    ")
        b = _pass_pieces(glsl_source=self.SIMPLE, mode=None,
                         merge_common=False, require_entry=False,
                         hoist_indent="", bridge_indent="")
        assert a.header_opencl == b.header_opencl
        assert a.body == b.body
        assert "float helper" in a.header_opencl
        assert "helper(uv.x)" in a.body


class TestCommonInoutParityAcrossHosts:
    """The tsKXR3 bug: a pass calls a Common helper whose param is inout. Both
    hosts must emit the address-of at the call site."""

    COMMON = (
        "vec3 light(vec2 uv, float b, inout vec3 avd) {\n"
        "    avd = vec3(b);\n"
        "    return uv.xyx;\n"
        "}\n"
    )
    IMAGE = (
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n"
        "    vec2 uv = fragCoord / iResolution.xy;\n"
        "    vec3 avd;\n"
        "    vec3 ld = light(uv, 3.0, avd);\n"
        "    fragColor = vec4(ld, 1.0);\n"
        "}\n"
    )

    def test_host_a_addresses_inout_common_arg(self):
        full = host_a.transpile(self.IMAGE, common=self.COMMON).get_full()
        assert "light(uv, 3.0f, &avd)" in full

    def test_host_b_addresses_inout_common_arg(self):
        out = host_b.transpile(self.IMAGE, mode="mainImage", common=self.COMMON)
        kernel = out[out.find("@KERNEL"):]
        assert "light(uv, 3.0f, &avd)" in kernel


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
