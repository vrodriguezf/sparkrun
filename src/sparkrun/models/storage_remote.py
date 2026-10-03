"""Local/SSH sessions for the standalone model storage worker."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import threading

from sparkrun.models.storage_worker import BACKEND, PROTOCOL
from sparkrun.orchestration.ssh import build_ssh_cmd, run_remote_script, should_run_locally
from sparkrun.utils.shell import quote


class WorkerError(RuntimeError):
    def __init__(self, message: str, code: str = "StorageUnreachable"):
        super().__init__(message)
        self.code = code


class WorkerSession:
    """A bounded JSON-lines session; EOF releases every target-side flock.

    Code goes over stdin to bash, then exec replaces bash with the worker. No
    remote source files, credentials in argv, global hf executable or SparkRun
    installation are required. Requests are sent only after the exec line.
    """

    def __init__(self, host: str, hf_home: str, ssh_kwargs: dict | None = None, *, timeout: float = 120):
        self.host, self.hf_home = host, hf_home
        self.ssh_kwargs = ssh_kwargs or {}
        self.timeout = timeout
        self.process = None
        self.responses = Queue()

    def __enter__(self):
        source = Path(__file__).with_name("storage_worker.py").read_bytes()
        bootstrap = "# SPARKRUN_STORAGE_WORKER\nimport base64;exec(compile(base64.b64decode(%r), '<sparkrun-storage>', 'exec'))" % (
            base64.b64encode(source).decode(),
        )
        local = should_run_locally(self.host, self.ssh_kwargs.get("ssh_user"))
        if local:
            command = [sys.executable, "-u", "-c", bootstrap]
        else:
            command = [*build_ssh_cmd(self.host, **self.ssh_kwargs), "bash", "-s"]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        assert self.process.stdin is not None and self.process.stdout is not None
        stdout = self.process.stdout
        if not local:
            script = (
                'storage_python="$HOME/.cache/sparkrun/model-storage/hf-%s/bin/python"; '
                'if [ ! -x "$storage_python" ]; then storage_python=python3; fi; '
                'exec "$storage_python" -u -c %s\n'
            ) % (BACKEND, quote(bootstrap))
            self.process.stdin.write(script.encode())
            self.process.stdin.flush()

        def reader():
            try:
                while line := stdout.readline(32 * 1024 * 1024 + 1):
                    self.responses.put(line)
            finally:
                self.responses.put(None)

        threading.Thread(target=reader, daemon=True).start()
        return self

    def request(self, action: str, **kwargs):
        assert self.process is not None and self.process.stdin is not None
        payload = {"schema_version": PROTOCOL, "action": action, "hf_home": self.hf_home, **kwargs}
        try:
            self.process.stdin.write((json.dumps(payload) + "\n").encode())
            self.process.stdin.flush()
            line = self.responses.get(timeout=self.timeout)
            if line is None:
                raise WorkerError("%s: storage helper disconnected; run sparkrun setup model-storage" % self.host)
            if len(line) > 32 * 1024 * 1024:
                raise WorkerError("%s: storage response exceeded size limit" % self.host)
            response = json.loads(line)
        except (BrokenPipeError, OSError, Empty, ValueError) as exc:
            raise WorkerError("%s: storage helper failed or timed out" % self.host) from exc
        if not response.get("ok"):
            error = response.get("error", {})
            raise WorkerError("%s: %s" % (self.host, error.get("message", "invalid response")), error.get("code", "StorageUnreachable"))
        return response["result"]

    def __exit__(self, *exc):
        if self.process:
            assert self.process.stdin is not None and self.process.stdout is not None
            try:
                self.process.stdin.close()
            except OSError:
                # A disconnected worker can leave buffered request bytes.
                # Closing must not mask its error or discard partial results.
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process.stdout.close()


def query(host: str, hf_home: str, action: str, *, ssh_kwargs: dict | None = None, **kwargs):
    with WorkerSession(host, hf_home, ssh_kwargs) as worker:
        return worker.request(action, **kwargs)


def setup_helper(host: str, ssh_kwargs: dict | None = None, *, offline: bool = False) -> None:
    """Explicitly prepare a pinned private helper. UV_* index/wheel settings work."""
    if should_run_locally(host, (ssh_kwargs or {}).get("ssh_user")):
        from sparkrun.models.storage_worker import StorageError, backend

        try:
            backend()
        except StorageError as exc:
            raise WorkerError(str(exc), exc.code) from exc
        return
    from sparkrun.core.tooling import UV_INSTALL_URL

    bootstrap = "" if offline else "command -v uv >/dev/null 2>&1 || curl -LsSf %s | sh\n" % quote(UV_INSTALL_URL)
    check = quote("import huggingface_hub as h; assert h.__version__ == %r" % BACKEND)
    script = (
        'set -eu\nexport PATH="$HOME/.local/bin:$PATH"\n'
        'storage_env="$HOME/.cache/sparkrun/model-storage/hf-%s"\n'
        'if [ -x "$storage_env/bin/python" ]; then\n'
        '  "$storage_env/bin/python" -c %s && exit 0\n'
        "elif python3 -c %s 2>/dev/null; then exit 0; fi\n"
    ) % (BACKEND, check, check)
    script += bootstrap + ('uv venv --allow-existing "$storage_env"\nuv pip install %s --python "$storage_env/bin/python" %s\n') % (
        "--offline" if offline else "",
        quote("huggingface-hub==" + BACKEND),
    )
    result = run_remote_script(host, script, **(ssh_kwargs or {}), timeout=600, session_guard=True)
    if not result.success:
        raise WorkerError("%s: helper setup failed: %s" % (host, result.stderr[-2000:]))
