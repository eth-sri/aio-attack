import datetime
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from time import sleep

from openai import OpenAI

from pruning_backdoor.helper.model import detect_model_fullpath


def _fix_extra_special_tokens_format(model_dir: str) -> None:
    """src/train.py saves its tokenizer via tokenizer.save_pretrained()
    under this venv's transformers (5.5.1), which writes a NEW
    'extra_special_tokens' key into tokenizer_config.json as a plain list of
    token strings (e.g. ['<|im_start|>', ...]). The base Qwen2.5-7B-Instruct
    checkpoint's own tokenizer_config.json never had this key at all. The
    vLLM docker image (ghcr.io/lambdalabsml/vllm-builder:v0.10.0) bundles an
    older transformers whose Qwen2 fast-tokenizer __init__ passes that value
    straight into _set_model_specific_special_tokens(), which unconditionally
    calls .keys() on it expecting a {name: token} dict -- AttributeError:
    'list' object has no attribute 'keys'. Since the base checkpoint works
    fine with no 'extra_special_tokens' key at all, the safe fix is to drop
    it entirely from any locally-saved model directory before serving it,
    rather than guess at a name mapping this list doesn't carry.
    """
    config_path = os.path.join(model_dir, "tokenizer_config.json")
    if not os.path.exists(config_path):
        return
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    if isinstance(config.get("extra_special_tokens"), list):
        del config["extra_special_tokens"]
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"[vllm_runner] dropped malformed list-typed 'extra_special_tokens' from {config_path} "
              "(vLLM's bundled transformers expects a dict here, matching the base checkpoint's own "
              "tokenizer_config.json, which never had this key)")


class VLLMRunner:
    """
    A context manager to run a vLLM server as a subprocess.
    The runner aggressively cleans the vLLM process tree on exit to avoid
    orphan worker/EngineCore processes holding GPU memory.
    """

    _serve_call_count = 0
    _serve_call_logmax = 10

    def __init__(
        self,
        model_name: str,
        serve_model_path: str | None = None,
        served_model_name: str | None = None,
        serve_extra_args: list[str] | None = None,
        logfile: str | None = None,
        hf_token: str | None = None,
        port: int = 8000,
        max_model_length: int = None,
        tensor_parallel_size: int = 1,
        data_parallel_size: int = 1,
        generation_config: str = "vllm",
        trials: int = 30,
        initial_sleep: int = 30,
        sleep_interval: int = 30,
        gpu_memory_utilization: float = 0.7,
        max_num_seqs: int = 128,
        vllm_docker_version: str = "v0.10.0",  # or latest
    ) -> None:
        if os.path.exists(model_name) and os.path.exists(os.path.join(model_name, "config.json")):
            self.use_local_model = True
            self.model_name = os.path.abspath(model_name)
        else:
            self.use_local_model = False
            self.model_name = model_name
        self.serve_model_path = os.path.abspath(serve_model_path) if serve_model_path else None
        self.served_model_name = served_model_name
        self.api_model_name = str(served_model_name) if served_model_name else str(self.model_name)
        self.serve_extra_args = list(serve_extra_args) if serve_extra_args else []
        self.hf_token = hf_token
        self.port = port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        self.max_model_length = max_model_length
        self.tensor_parallel_size = tensor_parallel_size
        self.data_parallel_size = data_parallel_size
        self.generation_config = generation_config
        self.process = None
        self.test_client = None
        self.trials = trials
        self.initial_sleep = initial_sleep
        self.sleep_interval = sleep_interval
        self.logfile = logfile
        self.vllm_docker_version = vllm_docker_version
        self._logfile_handle = None
        self.pid = None
        self.proc = None
        self._logfiles_base_dir = Path(__file__).parent.parent.parent / "vllm_logs"
        self._logfiles_base_dir.mkdir(parents=True, exist_ok=True)
        if self.logfile is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_model_name = self.model_name.replace("/", "_").replace(":", "_").replace("-", "_")
            # long staged/local model paths (deep dirs, long run names) can push the
            # derived filename past the filesystem's 255-byte name limit -> OSError
            # "File name too long" when opening it. Cap the stem and disambiguate
            # with a short hash of the full name instead of truncating silently.
            max_stem_len = 150
            if len(safe_model_name) > max_stem_len:
                digest = hashlib.sha1(safe_model_name.encode()).hexdigest()[:10]
                safe_model_name = f"{safe_model_name[-max_stem_len:]}_{digest}"
            self.logfile = str(self._logfiles_base_dir / f"vllm_{safe_model_name}_{timestamp}.log")
        print(f"vLLM log file will be saved at: {self.logfile}")
        self.is_vllm_available = shutil.which("vllm") is not None
        self.is_docker_available = shutil.which("docker") is not None
        if not self.is_vllm_available and not self.is_docker_available:
            msg = "Neither 'vllm' CLI nor 'docker' is available on PATH. Please install one of them."
            print(msg)
            if self._logfile_handle:
                self._logfile_handle.write(msg + "\n")
                self._logfile_handle.flush()
            raise FileNotFoundError(msg)

    def _collect_descendants(self, root_pid: int) -> list[int]:
        """
        Collect descendant PIDs via `pgrep -P` recursion.
        """
        seen = set()
        stack = [root_pid]
        out = []
        while stack:
            pid = stack.pop()
            try:
                res = subprocess.run(
                    ["pgrep", "-P", str(pid)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
            except Exception:
                continue
            if res.returncode != 0:
                continue
            for token in res.stdout.split():
                try:
                    child = int(token.strip())
                except Exception:
                    continue
                if child <= 0 or child in seen:
                    continue
                seen.add(child)
                out.append(child)
                stack.append(child)
        return out

    def _kill_process_tree(self, root_pid: int) -> None:
        """
        Best-effort kill for root pid and all descendants.
        """
        pids = self._collect_descendants(root_pid)
        # Kill children first, then root.
        kill_order = list(reversed(pids)) + [root_pid]
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for pid in kill_order:
                try:
                    os.kill(pid, sig)
                except Exception:
                    pass
            time.sleep(0.5)

    def _get_available_port(self, start_port: int) -> int:
        """
        Safely pick an available localhost port with interprocess synchronization and a short-lived reservation.
        This reduces race conditions where concurrent processes pick the same port.
        """
        import socket

        lock_path = "/tmp/vllm_port_select.lock"
        registry_path = "/tmp/vllm_port_registry.json"
        ttl_secs = 300  # reservation TTL (seconds)

        # Acquire a file lock to serialize selection across processes
        with open(lock_path, "w") as lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)
            # Load or initialize reservation registry
            try:
                with open(registry_path) as rf:
                    registry = json.load(rf)
            except Exception:
                registry = {}

            # Purge stale reservations (expired TTL or dead PID)
            now = time.time()
            to_delete = []
            for port_str, info in registry.items():
                try:
                    pid = int(info.get("pid", -1))
                except Exception:
                    pid = -1
                ts = float(info.get("ts", 0))
                if (now - ts) > ttl_secs:
                    to_delete.append(port_str)
                elif pid > 0:
                    try:
                        os.kill(pid, 0)  # check if process exists
                    except OSError:
                        to_delete.append(port_str)
            for k in to_delete:
                registry.pop(k, None)

            port = start_port
            chosen = None
            while True:
                # Skip reserved ports
                if str(port) not in registry:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        try:
                            s.bind(("0.0.0.0", port))
                            chosen = port
                            break
                        except OSError:
                            pass
                port += 1

            # Reserve the chosen port for this process
            registry[str(chosen)] = {"pid": os.getpid(), "ts": now}
            try:
                with open(registry_path, "w") as wf:
                    json.dump(registry, wf)
            except Exception:
                pass

            # Release the lock by exiting the context
        return chosen

    def _release_port_reservation(self, port: int) -> None:
        """Release a previously reserved port for this process."""
        lock_path = "/tmp/vllm_port_select.lock"
        registry_path = "/tmp/vllm_port_registry.json"
        try:
            with open(lock_path, "w") as lockf:
                fcntl.flock(lockf, fcntl.LOCK_EX)
                try:
                    with open(registry_path) as rf:
                        registry = json.load(rf)
                except Exception:
                    registry = {}
                key = str(port)
                info = registry.get(key)
                if isinstance(info, dict) and info.get("pid") == os.getpid():
                    registry.pop(key, None)
                    try:
                        with open(registry_path, "w") as wf:
                            json.dump(registry, wf)
                    except Exception:
                        pass
        except Exception:
            pass

    def _get_available_container_name(self) -> str:
        # PID-suffixed to avoid a check-then-create race when multiple tasks
        # start containers concurrently on the same node (listing existing
        # names first and picking a "free" one is not atomic).
        return f"vllm_docker_{os.getpid()}"

    def _wait_for_free_gpu0(self, threshold: float = 0.5, interval: int = 5, max_trial: int = 5) -> None:
        """
        Poll the first visible GPU's memory via nvidia-smi and wait until it is at or below the threshold.
        The "first visible GPU" is determined from CUDA_VISIBLE_DEVICES (first entry), falling back to 0 if unset.
        - threshold: fraction of total memory (e.g., 0.5 means 50%)
        - interval: seconds to wait between checks
        """
        if shutil.which("nvidia-smi") is None:
            print("nvidia-smi not found; skipping GPU memory check.")
            return

        # Determine first visible GPU (by CUDA_VISIBLE_DEVICES) or default to 0
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is None:
            target_gpu = "0"
        else:
            cvd = cvd.strip()
            if cvd == "":
                print("CUDA_VISIBLE_DEVICES is empty; skipping GPU memory check.")
                return
            target_gpu = cvd.split(",")[0].strip()
        query_cmd = ["nvidia-smi", "-i", target_gpu, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]

        n = 0
        last_ratio = None
        while n < max_trial:
            try:
                res = subprocess.run(query_cmd, capture_output=True, text=True, check=True)
                line = res.stdout.strip().splitlines()[0] if res.stdout.strip() else ""
                if not line:
                    # No output; do not block launch
                    print("nvidia-smi returned no output; skipping GPU memory wait.")
                    break
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 2:
                    print(f"Unexpected nvidia-smi output: {line!r}; skipping GPU memory wait.")
                    break
                used = float(parts[0])
                total = float(parts[1])
                ratio = used / total if total > 0 else 0.0
                last_ratio = ratio

                if ratio <= threshold:
                    # Sufficiently free
                    print(f"GPU {target_gpu} memory usage OK: {int(used)}/{int(total)} MiB ({ratio * 100:.1f}%) ≤ {int(threshold * 100)}%.")
                    break

                print(f"GPU {target_gpu} busy: {int(used)}/{int(total)} MiB ({ratio * 100:.1f}%) > {int(threshold * 100)}%. Waiting {interval}s...")
                sleep(interval)
                n += 1
            except Exception as e:
                print(f"Warning: failed to query GPU {target_gpu} memory with nvidia-smi ({e}); proceeding without wait.")
                break
        if last_ratio is not None and n >= max_trial and last_ratio > threshold:
            print(f"GPU {target_gpu} appears busy after maximum wait attempts; the runner may fail to start.")

    def _check_online(self) -> bool:
        self.test_client = OpenAI(
            api_key="dull-key",
            base_url=f"http://localhost:{self.port}/v1",
            timeout=600,
        )
        try:
            # Prefer a plain completion probe so models without chat templates
            # can still be recognized as online.
            _ = self.test_client.completions.create(
                model=self.api_model_name,
                prompt="Hello",
                timeout=30,
                temperature=0.0,
                max_tokens=1,
            )
            return True
        except Exception:
            pass

        try:
            _ = self.test_client.chat.completions.create(
                model=self.api_model_name,
                n=1,
                messages=[{"role": "user", "content": "Hello"}],
                timeout=30,
                max_completion_tokens=2,
            )
            return True
        except Exception as e:
            msg = str(e).lower()
            # Some base models intentionally lack chat templates; this still
            # indicates the server is up and should not block startup.
            if "chattemplate" in msg or "chat template" in msg:
                return True
            return False

    def __enter__(self) -> "VLLMRunner":
        def handle_sigint(signum, frame):
            # for capturing KeyboardInterrupt
            print(f"Received signal {signum}, shutting down vLLM server...")
            self.__exit__(None, None, None)
            sys.exit(0)

        signal.signal(signal.SIGINT, handle_sigint)
        self.port = self._get_available_port(self.port)
        if self.use_local_model:
            _fix_extra_special_tokens_format(self.model_name)
        # Wait until first visible GPU's memory usage is at or below 20% before launching vLLM
        self._wait_for_free_gpu0(threshold=1 - self.gpu_memory_utilization, interval=5)

        serve_model = self.serve_model_path or self.model_name
        vllm_serve_cfg = [serve_model, "--trust-remote-code"]
        if self.tensor_parallel_size:
            vllm_serve_cfg += ["--tensor-parallel-size", str(self.tensor_parallel_size)]
        if self.data_parallel_size:
            vllm_serve_cfg += ["--data-parallel-size", str(self.data_parallel_size)]
        if self.max_model_length:
            vllm_serve_cfg += ["--max-model-len", str(self.max_model_length)]
        if self.port:
            vllm_serve_cfg += ["--port", str(self.port)]
        if self.gpu_memory_utilization:
            vllm_serve_cfg += ["--gpu_memory_utilization", str(self.gpu_memory_utilization)]
        if self.generation_config:
            vllm_serve_cfg += ["--generation-config", self.generation_config]
        if self.max_num_seqs:
            vllm_serve_cfg += ["--max-num-seqs", str(self.max_num_seqs)]
        if self.served_model_name:
            vllm_serve_cfg += ["--served-model-name", str(self.served_model_name)]
        if self.serve_extra_args:
            vllm_serve_cfg += [str(x) for x in self.serve_extra_args]

        # Decide runtime mode: prefer local vLLM CLI if available, otherwise fallback to Docker
        if self.is_vllm_available:
            cmd = ["vllm", "serve"] + vllm_serve_cfg

        elif self.is_docker_available:
            self.container_name = self._get_available_container_name()

            cmd = [
                "docker",
                "run",
                "--name",
                self.container_name,
                "--gpus",
                "all",
                "-p",
                f"{self.port}:{self.port}",
                "--rm",
            ]
            if self.hf_token:
                cmd.extend(
                    [
                        "-e",
                        f"HUGGING_FACE_HUB_TOKEN={self.hf_token}",
                        "-e",
                        f"HF_TOKEN={self.hf_token}",
                    ]
                )
            # mount if it is a local model
            if self.use_local_model:
                # assert absname
                cmd.extend(["-v", f"{self.model_name}:{self.model_name}"])
            if self.serve_model_path and os.path.exists(self.serve_model_path):
                serve_model_path = os.path.abspath(self.serve_model_path)
                cmd.extend(["-v", f"{serve_model_path}:{serve_model_path}"])
            cmd += [f"ghcr.io/lambdalabsml/vllm-builder:{self.vllm_docker_version}"]
            cmd += vllm_serve_cfg

        print(f"Starting vLLM with command: {' '.join(cmd)}")
        # Increment global serve call counter and decide logging behavior
        VLLMRunner._serve_call_count += 1
        should_log = VLLMRunner._serve_call_count <= VLLMRunner._serve_call_logmax
        if should_log:
            self._logfile_handle = open(self.logfile, "a")
        if VLLMRunner._serve_call_count == VLLMRunner._serve_call_logmax:
            self._logfile_handle.write("Log is omitted from next call\n")
            self._logfile_handle.flush()
        stdout_target = self._logfile_handle if should_log else subprocess.DEVNULL
        stderr_target = subprocess.STDOUT if should_log else subprocess.DEVNULL
        child_env = os.environ.copy()
        if self.hf_token:
            child_env["HUGGING_FACE_HUB_TOKEN"] = self.hf_token
            child_env["HF_TOKEN"] = self.hf_token
        self.proc = subprocess.Popen(
            cmd,
            stdout=stdout_target,
            stderr=stderr_target,
            env=child_env,
            shell=False,
        )
        self.pid = self.proc.pid
        sleep(self.initial_sleep)  # give it some time to start
        # check if at least the process is running
        if self.proc.poll() is not None:
            # Release reserved port since the process failed to start
            self._release_port_reservation(self.port)
            self.__exit__(None, None, None)
            raise Exception(f"vLLM process with PID {self.pid} terminated unexpectedly, see log at {self.logfile} for details")
        # now check if the model is actually online
        for trial in range(self.trials):
            if self._check_online():
                print(f"vLLM is online after {trial * self.sleep_interval + self.initial_sleep} seconds")
                # Server is now bound to the port; release our reservation record
                self._release_port_reservation(self.port)
                return self
            else:
                print(f"vLLM not online yet, waiting {self.sleep_interval} seconds (trial {trial + 1}/{self.trials})...")
                sleep(self.sleep_interval)

        # Timed out waiting for server; release reservation and clean up
        self._release_port_reservation(self.port)
        self.__exit__(None, None, None)
        raise Exception(f"vLLM failed to start in time, see log at {self.logfile} for details")

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if getattr(self, "_exit_called", False):
            return
        self._exit_called = True

        if self.proc and self.proc.poll() is None:
            if self.is_vllm_available:
                try:
                    self._kill_process_tree(self.pid)
                    self.proc.wait(timeout=10)
                    print(f"vLLM process tree rooted at PID {self.pid} terminated.")
                except Exception as e:
                    print(f"Failed to terminate vLLM process tree rooted at PID {self.pid} gracefully: {e}.")
                    try:
                        self._kill_process_tree(self.pid)
                        self.proc.wait(timeout=5)
                        print(f"vLLM process tree rooted at PID {self.pid} killed")
                    except Exception as e:
                        print(
                            f"Failed to kill vLLM process tree rooted at PID {self.pid}: {e}. "
                            "You might need to kill it manually."
                        )
            elif self.is_docker_available:
                print(f"Terminating vLLM container {self.container_name}")
                try:
                    subprocess.run(
                        ["docker", "stop", self.container_name],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=self._logfile_handle if self._logfile_handle else subprocess.DEVNULL,
                    )
                except Exception as e:
                    print(f"Failed to stop docker container {self.container_name}: {e}. You might need to stop it manually. PID: {self.pid}")

        # Ensure any reserved port entry is released (covers early interrupts/errors)
        try:
            self._release_port_reservation(self.port)
        except Exception:
            pass

        if self._logfile_handle:
            self._logfile_handle.close()
            print(f"vLLM log file saved at: {self.logfile}")


if __name__ == "__main__":
    # example usage
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model name or path",
    )
    args = parser.parse_args()
    with VLLMRunner(model_name=detect_model_fullpath(args.model_name), max_model_length=256) as runner:
        client = OpenAI(
            api_key="dull-key",
            base_url=f"http://localhost:{runner.port}/v1",
            timeout=600,
        )

        print("calling vLLM API...")
        response = client.chat.completions.create(
            model=runner.model_name,  # local model needs to be in absolute path
            messages=[{"role": "user", "content": "Hello, vLLM!"}],
            max_completion_tokens=128,
            temperature=0.0,
        )
        print("Response from vLLM:")
        print(response.choices[0].message.content)

        while True:
            uin = input("Enter your input: ")
            if uin.lower() == "exit":
                break
            response = client.chat.completions.create(
                model=runner.model_name,
                messages=[{"role": "user", "content": uin}],
                max_completion_tokens=128,
                temperature=0.0,  # for greedy decoding
            )
            print("Response from vLLM:")
            print(response.choices[0].message.content)
