SPCONV_ALGO = 'auto'                                # 'auto', 'implicit_gemm', 'native'
# The bundled flex_gemm wheel ships its CUDA hashmap kernels but none of its Triton
# kernels, so every algorithm except 'explicit_gemm' fails at its first kernel call.
FLEX_GEMM_ALGO = 'explicit_gemm'                    # 'explicit_gemm', 'implicit_gemm', 'implicit_gemm_splitk', 'masked_implicit_gemm', 'masked_implicit_gemm_splitk'
FLEX_GEMM_HASHMAP_RATIO = 2.0                       # Ratio of hashmap size to input size

