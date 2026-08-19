"""Official WorldArena 2.0 Track 3 ``Policy`` entry point."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, Optional


# The official loader can import this file directly by path, in which case this
# directory is not a Python package. Add only the adapter directory itself.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from wsa_memory_adapter import RuntimeSettings, WSAMemoryWorldArenaAdapter


class Policy:
    """WSA-Memory implementation of the official A-side legacy Policy API."""

    def __init__(self, config_path: Optional[str] = None):
        self.settings = RuntimeSettings.from_sources(config_path)
        self.adapter = WSAMemoryWorldArenaAdapter(self.settings)

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.adapter.metadata

    def reset(self, reset_info: Optional[Dict[str, Any]] = None) -> None:
        self.adapter.reset(reset_info)

    def infer(self, new_obs: Dict[str, Any]) -> Dict[str, Any]:
        return self.adapter.infer(new_obs)
