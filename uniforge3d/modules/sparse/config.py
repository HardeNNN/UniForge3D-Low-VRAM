from typing import Literal

CONV = 'flex_gemm' 
DEBUG = False
ATTN = 'flash_attn'

# Transient-buffer budget for the sparse convolution path: anything larger (the neighbor
# map, the convolution's own output) is staged through host RAM, and a residual that a
# caller keeps across a convolution is parked there too.
OFFLOAD_BYTES = 256 * 1024 * 1024

def __from_env():
    import os
    
    global CONV
    global DEBUG
    global ATTN
    
    env_sparse_conv_backend = os.environ.get('SPARSE_CONV_BACKEND')
    env_sparse_debug = os.environ.get('SPARSE_DEBUG')
    env_sparse_attn_backend = os.environ.get('SPARSE_ATTN_BACKEND')
    if env_sparse_attn_backend is None:
        env_sparse_attn_backend = os.environ.get('ATTN_BACKEND')

    if env_sparse_conv_backend not in (None, 'flex_gemm'):
        raise ValueError(
            'This RecGen inference package only includes SPARSE_CONV_BACKEND=flex_gemm; '
            f'got {env_sparse_conv_backend!r}'
        )
    if env_sparse_debug is not None:
        DEBUG = env_sparse_debug == '1'
    if env_sparse_attn_backend is not None and env_sparse_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3']:
        ATTN = env_sparse_attn_backend


__from_env()
    

def set_conv_backend(backend: Literal['flex_gemm']):
    global CONV
    if backend != 'flex_gemm':
        raise ValueError('RecGen only includes the flex_gemm sparse-convolution backend')
    CONV = backend

def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug

def set_attn_backend(backend: Literal['xformers', 'flash_attn']):
    global ATTN
    ATTN = backend
