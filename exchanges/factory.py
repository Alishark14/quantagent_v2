"""Exchange adapter factory: get_adapter(name) with singleton cache."""

from __future__ import annotations

from typing import Any

from exchanges.base import ExchangeAdapter


class ExchangeFactory:
    """Singleton-cached factory for exchange adapters."""

    _instances: dict[str, ExchangeAdapter] = {}
    _registry: dict[str, type[ExchangeAdapter]] = {}

    @classmethod
    def register(cls, name: str, adapter_class: type[ExchangeAdapter]) -> None:
        """Register an adapter class by name."""
        cls._registry[name] = adapter_class

    @classmethod
    def get_adapter(cls, name: str, **kwargs: Any) -> ExchangeAdapter:
        """Get or create a singleton adapter instance."""
        if name in cls._instances:
            return cls._instances[name]
        if name not in cls._registry:
            raise ValueError(f"Unknown exchange: {name}")
        instance = cls._registry[name](**kwargs)
        cls._instances[name] = instance
        return instance

    @classmethod
    def reset(cls) -> None:
        """Clear all cached instances and registrations."""
        cls._instances.clear()
        cls._registry.clear()
