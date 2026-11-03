"""Key loader (safe to commit). This file holds NO secrets.

Your actual keys live in `keys.local.py`, which is git-ignored and never
shipped. On first run, if `keys.local.py` does not exist, it is created for
you by copying `keys.local.py.example`; open it in any editor,
paste your keys between the quotes, save, and run again.

To add a key for a future data source, add one line to KEYS in
`keys.local.py` - anything you add becomes an environment variable visible to
every layer and tool.

"""
import os
import shutil
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_LOCAL = _ROOT / "keys.local.py"
_EXAMPLE = _ROOT / "keys.local.py.example"


def _ensure_local() -> None:
    """Create keys.local.py from the example template on first run."""
    if not _LOCAL.exists() and _EXAMPLE.exists():
        shutil.copyfile(_EXAMPLE, _LOCAL)
        print(
            f"Created {_LOCAL.name} from the template. Open it, paste your "
            f"ANTHROPIC_API_KEY between the quotes, save, and run again."
        )


def _load_keys() -> dict:
    """Import KEYS from keys.local.py without making it a hard package import."""
    _ensure_local()
    if not _LOCAL.exists():
        return {}
    namespace: dict = {}
    exec(compile(_LOCAL.read_text(), str(_LOCAL), "exec"), namespace)  # noqa: S102
    return dict(namespace.get("KEYS", {}))


def apply() -> None:
    """Push every non-empty key into the environment (keys.local.py wins)."""
    for k, v in _load_keys().items():
        if v:
            os.environ[k] = v


def require(name: str) -> None:
    if not os.environ.get(name):
        raise SystemExit(
            f"\n{name} is not set.\n"
            f"Open keys.local.py in the project folder, paste your key between "
            f"the quotes for {name}, save, and run again.\n"
            f"(If keys.local.py is missing, run `python run.py` once to create it.)\n"
        )
