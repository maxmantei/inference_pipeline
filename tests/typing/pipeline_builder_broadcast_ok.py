from __future__ import annotations

from typing import Literal

from typing_extensions import reveal_type

from inference_pipeline.configs import BaseBroadcastStageConfig, BaseProducerConfig
from inference_pipeline.fanout import BroadcastStage
from inference_pipeline.pipeline import PipelineBuilder
from inference_pipeline.stages import ProducerStage


class NumberSource(ProducerStage[int, BaseProducerConfig]):
    def produce(self, stop=None):  # type: ignore[override]
        raise NotImplementedError


class IntBroadcast(BroadcastStage[int]):
    pass


builder = PipelineBuilder("inference")
stream = builder.source(
    "frames",
    NumberSource,
    config=BaseProducerConfig(out_maxsize=8, out_timeout=0.01),
)

two_a, two_b = stream.broadcast(
    "two-way",
    IntBroadcast,
    n_outputs=2,
    config=BaseBroadcastStageConfig[int](
        n_outputs=2,
        out_maxsize=8,
        out_timeout=0.01,
    ),
)
reveal_type(two_a)
reveal_type(two_b)

config3: BaseBroadcastStageConfig[int] = BaseBroadcastStageConfig(
    n_outputs=3,
    out_maxsize=8,
    out_timeout=0.01,
)
n3: Literal[3] = 3

three_a, three_b, three_c = stream.broadcast(
    "three-way",
    IntBroadcast,
    n_outputs=n3,
    config=config3,
)
reveal_type(three_a)
reveal_type(three_b)
reveal_type(three_c)
