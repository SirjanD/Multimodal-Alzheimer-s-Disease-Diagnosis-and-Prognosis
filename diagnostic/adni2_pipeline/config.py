"""Configuration loading with paths resolved relative to the config file."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PipelineConfig:
    """Validated project configuration."""

    path: Path
    values: dict[str, Any]

    @property
    def root(self) -> Path:
        """Return the directory containing the configuration file."""

        return self.path.parent

    @property
    def output_dir(self) -> Path:
        """Return the configured output directory."""

        return self.resolve(self.values["output_dir"])

    @property
    def seed(self) -> int:
        """Return the global random seed."""

        return int(self.values["random_seed"])

    def resolve(self, value: str | Path) -> Path:
        """Resolve a project-relative path without requiring the current directory."""

        candidate = Path(value)
        return candidate.resolve() if candidate.is_absolute() else (self.root / candidate).resolve()

    def data_path(self, key: str) -> Path:
        """Resolve one path from the data section."""

        return self.resolve(self.values["data"][key])


def load_config(path: str | Path) -> PipelineConfig:
    """Load a JSON configuration file and check required top-level keys."""

    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        values = json.load(stream)
    required = {"data", "output_dir", "random_seed", "graph", "gnn", "cross_validation", "survival"}
    missing = sorted(required.difference(values))
    if missing:
        raise ValueError(f"Configuration is missing required sections: {missing}")
    return PipelineConfig(config_path, values)
