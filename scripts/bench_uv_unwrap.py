"""Split and time the parameterization step (`cumesh.CuMesh.uv_unwrap`).

The export stage spends most of its time inside `uv_unwrap`: CUDA chart clustering, then the
xatlas C++ library computing and packing charts, then a gather loop. Which part dominates, and
whether the xatlas part already uses more than one thread, decides whether that time can be
attacked at all. Wall time and process CPU time are reported per step: a step that uses N cores
shows roughly N times more CPU time than wall time.

Usage: python scripts\\bench_uv_unwrap.py [faces]
"""

import sys
import time

import torch

import cumesh


def torus(faces: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A watertight torus with about `faces` triangles.

    Watertight on purpose: a mesh with split seams (a UV sphere repeats its wrap column) barely
    merges into charts, which moves the cost from growing charts to packing hundreds of
    thousands of two-face charts and measures something the pipeline never sees.
    """
    sides = max(8, int((faces / 2) ** 0.5))
    major, minor = 1.0, 0.35
    u = torch.linspace(0, 2 * torch.pi, sides + 1, dtype=torch.float32)[:-1]
    v = torch.linspace(0, 2 * torch.pi, sides + 1, dtype=torch.float32)[:-1]
    grid_u, grid_v = torch.meshgrid(u, v, indexing="ij")
    radius = major + minor * torch.cos(grid_v)
    vertices = torch.stack(
        [radius * torch.cos(grid_u), radius * torch.sin(grid_u), minor * torch.sin(grid_v)],
        dim=-1,
    ).reshape(-1, 3)
    index = torch.arange(sides, dtype=torch.int32)
    row = index[:, None]
    col = index[None, :]
    corner = (row * sides + col).reshape(-1)
    wrapped_right = (row * sides + (col + 1) % sides).reshape(-1)
    below = ((row + 1) % sides * sides + col).reshape(-1)
    below_right = ((row + 1) % sides * sides + (col + 1) % sides).reshape(-1)
    faces_tensor = torch.cat(
        [
            torch.stack([corner, below, below_right], dim=-1),
            torch.stack([corner, below_right, wrapped_right], dim=-1),
        ]
    ).contiguous()
    return vertices, faces_tensor


class Clock:
    """Wall and process CPU seconds around one step."""

    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self.wall = time.perf_counter()
        self.cpu = time.process_time()
        return self

    def __exit__(self, *exc):
        wall = time.perf_counter() - self.wall
        cpu = time.process_time() - self.cpu
        self.wall, self.cpu = wall, cpu
        print(f"{self.label:>22}: {wall:7.2f} s wall {cpu:7.2f} s cpu ({cpu / max(wall, 1e-9):4.2f} cores)")


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA device")
        return 1
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000

    vertices, faces = torus(target)
    print(f"{vertices.shape[0]} vertices, {faces.shape[0]} faces")
    mesh = cumesh.CuMesh()
    mesh.init(vertices.cuda(), faces.cuda())

    with Clock("clustering (CUDA)") as clustering:
        mesh.compute_charts()
    _, _, chart_vmap, chart_faces, chart_vertex_offset, chart_face_offset = mesh.read_atlas_charts()
    chart_vertices = mesh.read()[0][chart_vmap].cpu()
    chart_faces = chart_faces.cpu()
    chart_vertex_offset = chart_vertex_offset.cpu()
    chart_face_offset = chart_face_offset.cpu()
    print(f"{len(chart_vertex_offset) - 1} charts, {chart_faces.shape[0]} faces in charts")

    atlas = cumesh.xatlas.Atlas()
    with Clock("atlas add_mesh loop") as adding:
        for index in range(len(chart_vertex_offset) - 1):
            atlas.add_mesh(
                chart_vertices[chart_vertex_offset[index]:chart_vertex_offset[index + 1]],
                chart_faces[chart_face_offset[index]:chart_face_offset[index + 1]]
                - chart_vertex_offset[index],
            )

    # The atlas holds every chart as its own mesh, so this one call is the whole chart pass.
    with Clock("xatlas compute_charts") as computing:
        atlas.compute_charts()
    with Clock("xatlas pack_charts") as packing:
        atlas.pack_charts()

    print(
        f"total {clustering.wall + adding.wall + computing.wall + packing.wall:.2f} s,"
        f" xatlas {computing.wall + packing.wall:.2f} s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
