"""Paths and environment settings shared by every pipeline stage."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASES_DIR = PROJECT_ROOT / "cases"
CACHE_DIR = PROJECT_ROOT / ".cache" / "http"
DATA_DIR = PROJECT_ROOT / "data"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
SCHEMAS_DIR = PROJECT_ROOT / "schemas"
TEMPLATES_DIR = PROJECT_ROOT / "fdaengine" / "report" / "templates"
LOGS_DIR = PROJECT_ROOT / "logs"


def _load_dotenv() -> None:
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if value.strip() and not os.environ.get(key.strip()):
            os.environ[key.strip()] = value.strip()


_load_dotenv()


def sec_user_agent() -> str | None:
    """SEC requires contact details in the User-Agent; we never invent them."""
    return os.environ.get("SEC_USER_AGENT") or None


def openfda_api_key() -> str | None:
    return os.environ.get("OPENFDA_API_KEY") or None


def claude_exe() -> str | None:
    explicit = os.environ.get("CLAUDE_EXE")
    if explicit and Path(explicit).exists():
        return explicit
    default = Path(os.environ.get("USERPROFILE", Path.home())) / ".local" / "bin" / "claude.exe"
    if default.exists():
        return str(default)
    return shutil.which("claude")
