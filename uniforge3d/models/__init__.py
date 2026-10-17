"""UniForge3D model architectures and local checkpoint loading."""

import importlib

_MODEL_MODULES = {
    "SparseStructureDecoder": "vaes.sparse_structure",
    "SparseStructureFlowModel": "sparse_structure_flow",
    "JointDiTVGGTStage1Model": "joint_dit_vggt_stage1",
    "SLatFlowModel": "structured_latent_flow",
    "RoutedFullOnlySLatFlowModel": "structured_latent_flow",
    "FlexiDualGridVaeDecoder": "vaes.flexi_dual_grid",
    "SparseUnetVaeDecoder": "vaes.sparse_unet",
}

__all__ = list(_MODEL_MODULES)


def __getattr__(name: str):
    if name not in _MODEL_MODULES:
        raise AttributeError(f"module {__name__} has no attribute {name}")
    module = importlib.import_module(f".{_MODEL_MODULES[name]}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
