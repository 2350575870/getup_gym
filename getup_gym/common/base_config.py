"""Base configuration classes for getup_gym."""

from dataclasses import dataclass, fields, asdict
from collections.abc import Mapping
from typing import Any, Dict, Iterator


@dataclass
class BaseConfig(Mapping):
    """Base config class with dict-like behavior."""

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        for field in fields(self):
            yield field.name

    def __len__(self) -> int:
        return len(fields(self))

    def keys(self):
        return [field.name for field in fields(self)]

    def values(self):
        return [getattr(self, field.name) for field in fields(self)]

    def items(self):
        return [(field.name, getattr(self, field.name)) for field in fields(self)]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
