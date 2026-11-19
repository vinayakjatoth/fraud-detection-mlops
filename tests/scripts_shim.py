"""Import helper so tests can load scripts/update_readme_results.py."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "update_readme_results",
    Path(__file__).resolve().parents[1] / "scripts" / "update_readme_results.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
build_table, update_readme = _mod.build_table, _mod.update_readme
