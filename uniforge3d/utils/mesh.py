"""Shape-only remeshing and the established 512-to-1024 surface-voxel handoff."""

from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path
from typing import Any

import cumesh
import numpy as np
import torch
import trimesh

# Dual contouring only sets the density of the shell a stage exports, and that shell is
# decimated immediately afterwards. The baked texture does not depend on it either: it
# samples the full-resolution attribute volume and projects onto the original mesh. Raising
# this trades export time and VRAM for finer shell detail: 1024 measured 289s and 10.6GB on
# a case whose shell was 98% discarded, and it pushed the Shape-512 stage past the 8GB card
# so the driver spilled to host memory. 512 is what the vendored `o_voxel` exporter uses.
MAX_DC_RESOLUTION = 512


def export_glb(mesh: trimesh.Trimesh, path: Path, *, overwrite: bool) -> Path:
    """Atomically publish one complete GLB without truncating an older result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".glb", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        mesh.export(temporary, file_type="glb", extension_webp=True)
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def official_shape_only_postprocess(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    grid_size: int,
    pre_simplify_target: int,
    decimation_target: int,
    remesh_band: float,
    remesh_project: float,
) -> trimesh.Trimesh:
    """TRELLIS.2 example.py/to_glb geometry path without UV or texture work."""
    mesh = cumesh.CuMesh()
    mesh.init(vertices, faces)
    mesh.simplify(pre_simplify_target, verbose=False)
    try:
        mesh.fill_holes(max_hole_perimeter=3e-2)
    except RuntimeError as error:
        if "[CuMesh]" not in str(error) or "invalid configuration argument" not in str(error):
            raise
        warnings.warn(
            "CuMesh fill_holes rejected this mesh; continuing from the simplified mesh.",
            RuntimeWarning,
        )
        del mesh
        torch.cuda.empty_cache()
        mesh = cumesh.CuMesh()
        mesh.init(vertices, faces)
        mesh.simplify(pre_simplify_target, verbose=False)
    filled_vertices, filled_faces = mesh.read()
    bvh = cumesh.cuBVH(filled_vertices, filled_faces)
    # This shell is consumed only by `surface_voxelize` at 64^3, which is far coarser than
    # either grid, and it is decimated straight afterwards, so the configured 1024 grid buys
    # no visible detail while it costs seconds and gigabytes. Cap it as the texture path does
    # and release the pre-remesh mesh before the remesh allocates its own buffers.
    resolution = min(int(grid_size), MAX_DC_RESOLUTION)
    scale = (resolution + 3 * remesh_band) / resolution
    del mesh
    torch.cuda.empty_cache()
    remesh_vertices, remesh_faces = cumesh.remeshing.remesh_narrow_band_dc(
        filled_vertices,
        filled_faces,
        center=torch.zeros(3, device=vertices.device, dtype=vertices.dtype),
        scale=scale,
        resolution=resolution,
        band=remesh_band,
        project_back=remesh_project,
        verbose=False,
        bvh=bvh,
    )
    mesh = cumesh.CuMesh()
    mesh.init(remesh_vertices, remesh_faces)
    del remesh_vertices, remesh_faces
    mesh.simplify(decimation_target, verbose=False)
    mesh.compute_vertex_normals()
    output_vertices, output_faces = mesh.read()
    output_normals = mesh.read_vertex_normals()

    vertices_np = output_vertices.cpu().numpy()
    faces_np = output_faces.cpu().numpy()
    normals_np = output_normals.cpu().numpy()
    # Model coordinates -> GLB display coordinates: (x,y,z) -> (x,z,-y).
    vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
    normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2], -normals_np[:, 1]
    return trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces_np,
        vertex_normals=normals_np,
        process=False,
    )


def display_to_model(vertices: np.ndarray) -> np.ndarray:
    """Undo GLB export axes: (x,y,z) -> (x,-z,y)."""
    return np.asarray(vertices, dtype=np.float64)[:, [0, 2, 1]] * [1.0, -1.0, 1.0]


def normalize_to_cube(vertices: np.ndarray, target_max_side: float) -> tuple[np.ndarray, dict[str, Any]]:
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("Expected nonempty [N,3] vertices")
    if not np.isfinite(vertices).all() or not 0 < target_max_side < 1:
        raise ValueError("Nonfinite mesh or invalid target side")
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    center = (lower + upper) * 0.5
    longest_side = float((upper - lower).max())
    if longest_side <= 1e-12:
        raise ValueError("Degenerate mesh bounds")
    scale = target_max_side / longest_side
    normalized = (vertices - center) * scale
    bounds = np.stack((normalized.min(axis=0), normalized.max(axis=0)))
    if not np.allclose(bounds.mean(axis=0), 0.0, atol=1e-10):
        raise RuntimeError("Normalized mesh is not AABB centered")
    if not np.isclose(np.ptp(normalized, axis=0).max(), target_max_side, atol=1e-10):
        raise RuntimeError("Normalized mesh does not have the requested longest side")
    if np.abs(normalized).max() >= 0.5:
        raise RuntimeError("Normalized mesh is outside the open unit cube")
    transform = np.eye(4)
    transform[:3, :3] *= scale
    transform[:3, 3] = -center * scale
    return normalized, {
        "center_model": center.tolist(),
        "uniform_scale": scale,
        "target_max_side": target_max_side,
        "source_bounds_model": [lower.tolist(), upper.tolist()],
        "normalized_bounds_model": bounds.tolist(),
        "model_to_normalized_4x4": transform.tolist(),
    }


def surface_voxelize(
    vertices: np.ndarray,
    faces: np.ndarray,
    resolution: int,
    *,
    max_faces_per_chunk: int = 100_000,
) -> np.ndarray:
    """Voxelize triangle surfaces with an exact triangle-box SAT test."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if max_faces_per_chunk <= 0:
        raise ValueError("max_faces_per_chunk must be positive")

    half_size = 0.5 / resolution
    coord_chunks = []
    for start in range(0, len(faces), max_faces_per_chunk):
        triangles = vertices[faces[start : start + max_faces_per_chunk]]
        grid_triangles = (triangles + 0.5) * resolution
        lower = np.floor(grid_triangles.min(axis=1)).astype(np.int32)
        upper = np.floor(grid_triangles.max(axis=1)).astype(np.int32)
        lower = np.clip(lower, 0, resolution - 1)
        upper = np.clip(upper, 0, resolution - 1)
        dimensions = upper - lower + 1

        for dimensions_key in np.unique(dimensions, axis=0):
            triangle_ids = np.flatnonzero(np.all(dimensions == dimensions_key, axis=1))
            ranges = [np.arange(int(size), dtype=np.int32) for size in dimensions_key]
            offsets = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3)
            candidate_coords = lower[triangle_ids, None, :] + offsets[None, :, :]
            candidate_coords = candidate_coords.reshape(-1, 3)
            candidate_triangles = np.broadcast_to(
                triangles[triangle_ids, None, :, :],
                (len(triangle_ids), len(offsets), 3, 3),
            ).reshape(-1, 3, 3)
            centers = -0.5 + (candidate_coords.astype(np.float64) + 0.5) / resolution
            intersects = _triangles_intersect_axis_aligned_boxes(
                candidate_triangles,
                centers,
                half_size,
            )
            coord_chunks.append(candidate_coords[intersects].astype(np.int32, copy=False))

    coords = np.concatenate(coord_chunks, axis=0) if coord_chunks else np.empty((0, 3), dtype=np.int32)
    if coords.size == 0:
        raise RuntimeError("Surface voxelization returned no voxels")
    coords = np.unique(coords, axis=0)
    if int(coords.min()) < 0 or int(coords.max()) >= resolution:
        raise RuntimeError("Surface voxels are outside the canonical grid")
    return coords


def _triangles_intersect_axis_aligned_boxes(
    triangles: np.ndarray,
    centers: np.ndarray,
    half_size: float,
) -> np.ndarray:
    """Vectorized Akenine-Moller triangle/AABB overlap test."""
    local = triangles - centers[:, None, :]
    edge0 = local[:, 1] - local[:, 0]
    edge1 = local[:, 2] - local[:, 1]
    edge2 = local[:, 0] - local[:, 2]
    keep = np.ones(len(local), dtype=bool)
    epsilon = np.finfo(np.float64).eps * 32

    keep &= np.all(local.min(axis=1) <= half_size + epsilon, axis=1)
    keep &= np.all(local.max(axis=1) >= -half_size - epsilon, axis=1)

    for edge in (edge0, edge1, edge2):
        axes = (
            np.stack((np.zeros(len(edge)), edge[:, 2], -edge[:, 1]), axis=1),
            np.stack((-edge[:, 2], np.zeros(len(edge)), edge[:, 0]), axis=1),
            np.stack((edge[:, 1], -edge[:, 0], np.zeros(len(edge))), axis=1),
        )
        for axis in axes:
            projection = np.einsum("nvc,nc->nv", local, axis)
            radius = half_size * np.abs(axis).sum(axis=1) + epsilon
            keep &= projection.min(axis=1) <= radius
            keep &= projection.max(axis=1) >= -radius

    normal = np.cross(edge0, local[:, 2] - local[:, 0])
    plane_distance = np.einsum("nc,nc->n", normal, local[:, 0])
    plane_radius = half_size * np.abs(normal).sum(axis=1) + epsilon
    keep &= np.abs(plane_distance) <= plane_radius
    return keep
