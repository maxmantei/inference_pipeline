"""Base configuration dataclasses for stages."""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(slots=True, frozen=True, kw_only=True)
class BaseConfig:
    """Base configuration for stages.

    This class is frozen to ensure immutability of stage configuration at runtime.
    """

    name: str | None = None


@dataclass(slots=True, frozen=True, kw_only=True)
class BaseStageConfig(BaseConfig):
    """Base configuration for stages."""

    out_maxsize: int
    out_timeout: float


@dataclass(slots=True, frozen=True, kw_only=True)
class BaseProducerConfig(BaseStageConfig):
    """Configuration for producer stages."""

    pass


@dataclass(slots=True, frozen=True, kw_only=True)
class BaseProcessorConfig(BaseStageConfig):
    """Configuration for processor stages."""

    pass


@dataclass(slots=True, frozen=True, kw_only=True)
class BaseConsumerConfig(BaseConfig):
    """Configuration for consumer stages."""

    pass


@dataclass(slots=True, frozen=True, kw_only=True)
class BaseBroadcastStageConfig[T](BaseStageConfig):
    """Base configuration for broadcast stages."""

    n_outputs: int
    copy_fn: Callable[[T], T] | None = None


@dataclass(slots=True, frozen=True, kw_only=True)
class BaseSplitStageConfig(BaseConfig):
    """Base configuration for split stages."""

    out_a_maxsize: int
    out_b_maxsize: int
    out_a_timeout: float
    out_b_timeout: float
