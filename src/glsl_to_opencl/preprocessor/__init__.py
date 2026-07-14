"""
Preprocessor transformation module.

Handles GLSL preprocessor directives transformation to OpenCL equivalents.
"""

from .preprocessor_transformer import PreprocessorTransformer
from .uniform_redefine import (
    append_uniform_undefs,
    collect_uniform_redefines,
    find_redefined_uniforms,
    uniform_redefine_prefix,
)

__all__ = [
    'PreprocessorTransformer',
    'append_uniform_undefs',
    'collect_uniform_redefines',
    'find_redefined_uniforms',
    'uniform_redefine_prefix',
]
