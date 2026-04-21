from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from robustsep_pkg.core.artifact_io import canonical_json_hash, write_json


@dataclass
class RunManifest:
    run_id: str
    created_unix: float = field(default_factory=time.time)
    parameters: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["manifest_hash"] = canonical_json_hash(
            {k: v for k, v in data.items() if k != "manifest_hash"}
        )
        return data

    def write(self, path: str) -> None:
        write_json(path, self.to_dict())

    def record_parameter(self, key: str, value: Any) -> None:
        self.parameters[key] = value

    def record_artifact(self, key: str, value: Any) -> None:
        self.artifacts[key] = value

    def record_diagnostic(self, key: str, value: Any) -> None:
        self.diagnostics[key] = value
