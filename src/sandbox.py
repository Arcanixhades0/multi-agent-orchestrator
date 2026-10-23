"""
Runs LLM-written code.

NEVER exec() model output inside your own process: it could read files,
delete things, or hang forever. Two options here:

  docker      Runs in a throwaway container with no network, 256MB RAM, 1 CPU.
              This is real isolation.
  subprocess  A separate Python process with a timeout. Protects against hangs
              and crashes, but NOT against malicious code. Dev fallback only.
"""
import shutil
import subprocess
import sys
import warnings

from src.config import SANDBOX, SANDBOX_TIMEOUT

DOCKER_IMAGE = "python:3.12-slim"


def _docker_available():
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


_use_docker = None


def run_python(code, timeout=SANDBOX_TIMEOUT):
    global _use_docker

    if _use_docker is None:
        _use_docker = SANDBOX == "docker" and _docker_available()
        if SANDBOX == "docker" and not _use_docker:
            warnings.warn("Docker is not running: falling back to an UNISOLATED subprocess sandbox.")

    if _use_docker:
        cmd = ["docker", "run", "--rm", "-i", "--network", "none",
               "--memory", "256m", "--cpus", "1", DOCKER_IMAGE, "python", "-"]
    else:
        # -I: isolated mode, ignores env vars and user site-packages
        cmd = [sys.executable, "-I", "-"]

    try:
        # Explicit UTF-8: Windows defaults to cp1252, which crashes on characters like "→" in LLM code
        result = subprocess.run(cmd, input=code, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"passed": False, "stdout": "", "stderr": f"Timed out after {timeout}s"}

    return {
        "passed": result.returncode == 0,
        "stdout": result.stdout[-2000:],
        "stderr": result.stderr[-2000:],
    }
