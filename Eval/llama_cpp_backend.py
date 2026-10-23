"""Shared llama.cpp (GGUF) inference backend for the Eval/test_model_* scripts.

The three refusal / mcd / backdoor eval scripts format their prompts with the
model's chat template (exactly as for the vLLM backend) and then hand the
finished prompt *strings* to `generate_completions_llama_cpp`, which:

  * resolves a `.gguf` file (a file path, or the quantized `.gguf` inside a dir),
  * launches a `llama-server` subprocess bound to a free localhost port,
  * drives it through its OpenAI-compatible ``/v1/completions`` endpoint with a
    small thread pool (raw-prompt completions -- the template is already applied),
  * tears the whole server process group down on exit.

Environment overrides:
  LLAMA_CPP_SERVER_BIN   path to the llama-server binary
  LLAMA_CPP_MAX_WORKERS  concurrent in-flight requests (default: n_parallel)
  LLAMA_CPP_READY_TIMEOUT seconds to wait for the server to come up (default 300)
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path



def resolve_server_bin() -> str:
    for cand in (
        os.getenv("LLAMA_CPP_SERVER_BIN"),
        shutil.which("llama-server"),
    ):
        if cand and os.path.isfile(cand):
            return cand
    raise FileNotFoundError(
        "llama-server binary not found. Build llama.cpp or set LLAMA_CPP_SERVER_BIN."
    )


def resolve_gguf_file(model_path: str) -> str:
    """Return a concrete .gguf file path from a file or a directory.

    Inside a directory the quantized shard (e.g. ``*.Q4_K_M.gguf``) is preferred
    over an ``f16``/``f32``/``bf16`` conversion intermediate.
    """
    p = Path(model_path)
    if p.is_file() and p.suffix.lower() == ".gguf":
        return str(p)
    if p.is_dir():
        ggufs = sorted(p.glob("*.gguf"))
        quantized = [
            f for f in ggufs
            if not any(t in f.name.lower() for t in (".f16.", ".f32.", ".bf16."))
        ]
        chosen = quantized or ggufs
        if chosen:
            return str(chosen[0])
    raise FileNotFoundError(f"no .gguf file found at {model_path}")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class LlamaCppServer:
    """Context manager around a `llama-server` subprocess."""

    def __init__(
        self,
        gguf_file: str,
        *,
        ctx_size: int = 8192,
        n_gpu_layers: int = -1,
        n_parallel: int = 4,
        logfile: str | None = None,
        ready_timeout: int | None = None,
    ) -> None:
        self.gguf_file = gguf_file
        self.ctx_size = ctx_size
        self.n_gpu_layers = n_gpu_layers
        self.n_parallel = max(1, n_parallel)
        self.port = _free_port()
        self.alias = "gguf-model"
        self.proc: subprocess.Popen | None = None
        self._log = None
        self.ready_timeout = (
            ready_timeout
            if ready_timeout is not None
            else int(os.getenv("LLAMA_CPP_READY_TIMEOUT", "300"))
        )
        if logfile is None:
            logdir = Path(__file__).parent / "vllm_logs"
            logdir.mkdir(parents=True, exist_ok=True)
            logfile = str(logdir / f"llamacpp_{os.getpid()}_{self.port}.log")
        self.logfile = logfile

    def __enter__(self) -> "LlamaCppServer":
        cmd = [
            resolve_server_bin(),
            "-m", self.gguf_file,
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "-c", str(self.ctx_size),
            "-np", str(self.n_parallel),
            "-ngl", str(self.n_gpu_layers),
            "-a", self.alias,
            "--no-webui",
        ]
        self._log = open(self.logfile, "w")
        print(f"[llama_cpp] launching: {' '.join(cmd)}", flush=True)
        print(f"[llama_cpp] server log: {self.logfile}", flush=True)
        self.proc = subprocess.Popen(
            cmd, stdout=self._log, stderr=subprocess.STDOUT, preexec_fn=os.setsid
        )
        self._wait_until_ready()
        return self

    def _wait_until_ready(self) -> None:
        from openai import OpenAI

        client = OpenAI(
            api_key="dull-key",
            base_url=f"http://127.0.0.1:{self.port}/v1",
            timeout=60,
        )
        deadline = time.time() + self.ready_timeout
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited early (code {self.proc.returncode}); "
                    f"see {self.logfile}"
                )
            try:
                client.completions.create(
                    model=self.alias, prompt="hi", max_tokens=1, temperature=0.0
                )
                print(f"[llama_cpp] server ready on port {self.port}", flush=True)
                return
            except Exception:
                time.sleep(2)
        raise RuntimeError(
            f"llama-server not ready within {self.ready_timeout}s; see {self.logfile}"
        )

    def __exit__(self, *exc) -> bool:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                try:
                    self.proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        if self._log is not None:
            try:
                self._log.close()
            except Exception:
                pass
        return False


def generate_completions_llama_cpp(
    model_path: str,
    prompts: list[str],
    max_new_tokens: int,
    *,
    ctx_size: int = 8192,
    n_gpu_layers: int = -1,
    n_parallel: int = 4,
) -> list[str]:
    """Run greedy completion for every already-formatted prompt string."""
    from openai import OpenAI

    gguf_file = resolve_gguf_file(model_path)
    outputs: list[str] = [""] * len(prompts)
    max_workers = max(1, int(os.getenv("LLAMA_CPP_MAX_WORKERS", str(n_parallel))))

    with LlamaCppServer(
        gguf_file,
        ctx_size=ctx_size,
        n_gpu_layers=n_gpu_layers,
        n_parallel=n_parallel,
    ) as server:
        client = OpenAI(
            api_key="dull-key",
            base_url=f"http://127.0.0.1:{server.port}/v1",
            timeout=180,
        )

        def _infer_one(idx: int, prompt: str) -> tuple[int, str]:
            try:
                resp = client.completions.create(
                    model=server.alias,
                    prompt=str(prompt),
                    temperature=0.0,
                    max_tokens=max_new_tokens,
                )
                text = (resp.choices[0].text or "").strip() if resp.choices else ""
            except Exception as exc:  # noqa: BLE001 - mirror the vLLM backend's leniency
                print(
                    f"[llama_cpp warn] request failed at sample {idx + 1}/{len(prompts)}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                text = ""
            return idx, text

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_infer_one, i, p): i for i, p in enumerate(prompts)
            }
            for future in as_completed(futures):
                i, text = future.result()
                outputs[i] = text

    return outputs
