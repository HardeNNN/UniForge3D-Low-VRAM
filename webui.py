"""Gradio web UI for UniForge3D on an 8 GB card.

Runs the same pipeline as ``infer.py``. The pipeline keeps the completed stages resident,
so a warm job does not re-read them; the first job starts within seconds because the
weights load a stage at a time, with the later ones fetched while the GPU works. Jobs run
one at a time: that is all the 8 GB budget holds, and the queue makes the wait visible
instead of letting two cases race for the card.

    python webui.py                        # http://127.0.0.1:7860
    python webui.py --port 8000
    python webui.py --low-ram              # one stage resident at a time
    python webui.py --no-low-vram --weights-dtype float32

Every job writes its own directory under ``outputs/webui/``: a copy of the exact images
it ran on, plus ``prediction.glb`` and, when the box is ticked, ``textures/*.png``. The
copy is what makes a result reproducible later.

The page stays on the loopback interface. There is no share link: exposing a 24 GB
model stack on a public URL is a decision to make deliberately, not a flag on a demo.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import shutil
import threading
import time
from pathlib import Path

# The CUDA allocator must be configured before torch initialises its context, exactly as
# in infer.py. Windows builds do not implement expandable segments.
if os.name != "nt":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import torch  # noqa: E402
from uniforge3d import UniForge3DImageTo3DPipeline  # noqa: E402

try:
    import gradio as gr
except ImportError as error:  # pragma: no cover - install-time problem, not a code path
    raise SystemExit(
        f"gradio is not installed ({error}).\n"
        "Run scripts\\install_windows.ps1, or install it on its own:\n"
        "  .venv\\Scripts\\python.exe -m pip install \"gradio>=5,<6\""
    )

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "weights" / "pipeline.local.json"
LOG_TAIL = 400


class JobLog(io.TextIOBase):
    """A stdout/stderr sink that keeps the tail of a running job for the page.

    tqdm redraws one line with carriage returns, so those count as line breaks too and
    the page shows the newest state of each bar instead of every intermediate frame.
    """

    def __init__(self, limit: int = LOG_TAIL) -> None:
        self._limit = limit
        self._lines: list[str] = []

    def write(self, text: str) -> int:
        self._lines += [line for line in text.replace("\r", "\n").split("\n") if line.strip()]
        del self._lines[: max(0, len(self._lines) - self._limit)]
        return len(text)

    def text(self) -> str:
        return "\n".join(self._lines)


class Service:
    """The resident pipeline plus the output layout the page writes into."""

    def __init__(
        self,
        config: Path | None,
        model: str,
        output: Path,
        *,
        device: int,
        low_vram: bool,
        weights_dtype: str | None,
        low_ram: bool = False,
        prefetch: bool = True,
    ) -> None:
        self.config = config
        self.model = model
        self.output = output
        self.device = device
        self.low_vram = low_vram
        self.low_ram = low_ram
        self.prefetch = prefetch
        # 'auto' means "keep the precision the checkpoints were published with", which the
        # pipeline spells as None (infer.py does the same translation).
        self.weights_dtype = None if weights_dtype == "auto" else weights_dtype
        self._pipeline: UniForge3DImageTo3DPipeline | None = None
        self._load_lock = threading.Lock()

    def pipeline(self) -> UniForge3DImageTo3DPipeline:
        """Load on first use, so browsing the page costs nothing.

        Since the weights load a stage at a time, this returns in well under a second and
        the first job picks the loading up as it goes.
        """
        with self._load_lock:
            if self._pipeline is None:
                started = time.perf_counter()
                options = {
                    "device": self.device,
                    "low_vram": self.low_vram,
                    "weights_dtype": self.weights_dtype,
                    "low_ram": self.low_ram,
                    "prefetch": self.prefetch,
                }
                self._pipeline = (
                    UniForge3DImageTo3DPipeline.from_config(self.config, **options)
                    if self.config is not None
                    else UniForge3DImageTo3DPipeline.from_pretrained(self.model, **options)
                )
                print(f"[webui] pipeline ready in {time.perf_counter() - started:.1f}s", flush=True)
        return self._pipeline

    def case_dir(self, name: str | None) -> Path:
        """A fresh directory for one job; the name is optional and made path-safe."""
        stem = re.sub(r"[^A-Za-z0-9._-]+", "-", (name or "").strip()).strip("-._")
        stem = stem or time.strftime("job-%Y%m%d-%H%M%S")
        candidate, index = self.output / stem, 1
        while candidate.exists():
            index += 1
            candidate = self.output / f"{stem}-{index}"
        return candidate

    def generate(
        self,
        uploads: list[str] | None,
        seed: float | None,
        name: str | None,
        save_textures: bool,
    ):
        """Run one case, yielding (log, glb, download) until it finishes.

        A generator rather than a plain call so the page can show progress: a case takes
        about four minutes, and a silent spinner for four minutes reads as a hang.
        """
        if not uploads:
            yield "Load 2-4 images of one object first.", None, None
            return

        case = self.case_dir(name)
        inputs = case / "input"
        inputs.mkdir(parents=True)
        images: list[Path] = []
        for index, upload in enumerate(sorted(Path(path) for path in uploads), start=1):
            target = inputs / f"{index:03d}{upload.suffix.lower()}"
            shutil.copyfile(upload, target)
            images.append(target)
        output = case / "prediction.glb"

        log = JobLog()
        result: dict[str, object] = {}

        def work() -> None:
            try:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    if len(images) < 2:
                        print(
                            f"[webui] {len(images)} image: the model is built for sparse "
                            "multi-view input, so expect a guess rather than a reconstruction",
                            flush=True,
                        )
                    print(f"[webui] {case.name}: {len(images)} image(s) -> {output}", flush=True)
                    self.pipeline().run(
                        images,
                        output,
                        seed=None if seed is None else int(seed),
                        save_textures=bool(save_textures),
                    )
            except Exception as error:  # noqa: BLE001 - the page must survive any failure
                result["error"] = f"{type(error).__name__}: {error}"
            finally:
                result["done"] = True

        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        while not result.get("done"):
            time.sleep(0.5)
            yield log.text(), None, None
        if "error" in result:
            yield f"{log.text()}\n\n[webui] failed — {result['error']}", None, None
            return
        written = [output, *sorted((case / "textures").glob("*.png"))]
        files = "\n".join(f"[webui] wrote {path}" for path in written)
        yield f"{log.text()}\n\n{files}", str(output), str(output)


def build_ui(service: Service) -> gr.Blocks:
    source = str(service.config) if service.config is not None else service.model
    with gr.Blocks(title="UniForge3D", analytics_enabled=False) as demo:
        gr.Markdown(
            "# UniForge3D\n"
            "Images to 3D asset. Load **2-4 photos of one object** from different angles — "
            "the views are sparse and unposed, so order does not matter. "
            "Weights load one stage at a time, so a job starts within seconds and the rest "
            "of the stack arrives while the GPU works."
        )
        with gr.Row():
            with gr.Column():
                uploads = gr.Files(
                    label="Views of one object",
                    file_count="multiple",
                    file_types=["image"],
                )
                name = gr.Textbox(label="Case name (optional)")
                seed = gr.Number(label="Seed (optional)", precision=0)
                textures = gr.Checkbox(
                    label="Also save the baked texture maps as PNG", value=True
                )
                run = gr.Button("Generate", variant="primary")
                gr.Markdown(
                    f"Models: `{source}` · low-VRAM mode "
                    f"{'on' if service.low_vram else 'off'} · weights "
                    f"{'one stage at a time (low RAM)' if service.low_ram else 'prefetched'}"
                    f" · output under `{service.output}`"
                )
            with gr.Column():
                glb = gr.Model3D(
                    label="prediction.glb — the interactive view needs WebGPU enabled",
                    clear_color=[0.0, 0.0, 0.0, 0.0],
                )
                download = gr.File(label="Download", interactive=False)
                log = gr.Textbox(label="Progress", lines=16, max_lines=16, autoscroll=True)
        run.click(
            fn=service.generate,
            inputs=[uploads, seed, name, textures],
            outputs=[log, glb, download],
        )
    return demo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None, help="local pipeline config")
    parser.add_argument("--model", default="xibushu/UniForge3D", help="Hub repository, if not using --config")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "webui")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--weights-dtype",
        default=None,
        help="weight precision for the denoisers: float16 (default on --low-vram), "
             "float32, or 'auto' for whatever the residency mode uses anyway",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--open", action="store_true", help="open a browser on start")
    parser.add_argument(
        "--low-ram", action="store_true",
        help="keep only the stage being run in memory: peak RAM is one group of weights "
             "instead of the whole 25 GiB stack, and every job reloads them from disk",
    )
    parser.add_argument(
        "--prefetch", action=argparse.BooleanOptionalAction, default=True,
        help="load later stages' weights on a background thread while the GPU runs the "
             "current stage (never used by --low-ram)",
    )
    args = parser.parse_args()

    config = args.config
    if config is None and args.model == parser.get_default("model"):
        # The default path is the local one: this fork's whole point is that inference
        # never touches the Hub.
        config = DEFAULT_CONFIG
    if config is not None and not config.is_file():
        print(
            f"[webui] missing {config}\n"
            "[webui] run 'python scripts\\fetch.py weights' and 'python scripts\\make_local_config.py',\n"
            "[webui] or start with --model xibushu/UniForge3D",
            flush=True,
        )
        return 1

    if not torch.cuda.is_available():
        print("[webui] no CUDA device visible to torch", flush=True)
        return 1

    service = Service(
        config,
        args.model,
        args.output.resolve(),
        device=args.device,
        low_vram=args.low_vram,
        weights_dtype=args.weights_dtype,
        low_ram=args.low_ram,
        prefetch=args.prefetch,
    )
    # One job at a time is a property of the card, not a preference.
    build_ui(service).queue(default_concurrency_limit=1, max_size=8).launch(
        server_name=args.host, server_port=args.port, inbrowser=args.open
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
