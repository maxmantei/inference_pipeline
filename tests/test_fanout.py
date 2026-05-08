from __future__ import annotations

import copy
import threading

import pytest

from inference_pipeline.buffer import BufferQ
from inference_pipeline.configs import BaseBroadcastStageConfig, BaseSplitStageConfig
from inference_pipeline.fanout import BroadcastStage, SplitStage


class _ParitySplit(SplitStage[int, str, int, BaseSplitStageConfig]):
    def __init__(self, config: BaseSplitStageConfig | None = None) -> None:
        super().__init__(
            config=config
            or BaseSplitStageConfig(
                name="parity-split",
                out_a_maxsize=16,
                out_b_maxsize=16,
                out_a_timeout=0.01,
                out_b_timeout=0.01,
            )
        )

    def split(self, item: int) -> tuple[str | None, int | None]:
        if item < 0:
            return None, None
        if item % 2 == 0:
            return f"even:{item}", None
        return None, item


def test_broadcast_stage_rejects_zero_outputs() -> None:
    with pytest.raises(ValueError, match="n_outputs must be >= 1"):
        BroadcastStage[int](
            config=BaseBroadcastStageConfig(
                n_outputs=0,
                out_maxsize=4,
                out_timeout=0.01,
            )
        )


def test_broadcast_outputs_exposes_all_queues_consistently() -> None:
    stage = BroadcastStage[int](
        config=BaseBroadcastStageConfig(
            n_outputs=3,
            out_maxsize=4,
            out_timeout=0.01,
        )
    )

    outputs = stage.outputs

    assert isinstance(outputs, tuple)
    assert len(outputs) == 3
    assert outputs[0] is stage.output(0)
    assert outputs[1] is stage.output(1)
    assert outputs[2] is stage.output(2)


def test_broadcast_stage_aliases_payload_without_copy_fn() -> None:
    stage: BroadcastStage[dict[str, int]] = BroadcastStage(
        config=BaseBroadcastStageConfig(
            n_outputs=2,
            out_maxsize=8,
            out_timeout=0.01,
        )
    )
    payload = {"count": 1}
    input_q: BufferQ[dict[str, int]] = BufferQ(maxsize=8, default_timeout=0.01)
    input_q.put(payload)
    input_q.close()

    stage.run(input_q)

    out0 = list(stage.output(0).iter_until_closed())
    out1 = list(stage.output(1).iter_until_closed())
    assert len(out0) == 1
    assert len(out1) == 1
    assert out0[0] is payload
    assert out1[0] is payload
    assert out0[0] is out1[0]


def test_broadcast_stage_copy_fn_creates_independent_payloads() -> None:
    stage: BroadcastStage[dict[str, int]] = BroadcastStage(
        config=BaseBroadcastStageConfig(
            n_outputs=2,
            out_maxsize=8,
            out_timeout=0.01,
            copy_fn=copy.deepcopy,
        )
    )
    payload = {"count": 10}
    input_q: BufferQ[dict[str, int]] = BufferQ(maxsize=8, default_timeout=0.01)
    input_q.put(payload)
    input_q.close()

    stage.run(input_q)

    out0 = list(stage.output(0).iter_until_closed())
    out1 = list(stage.output(1).iter_until_closed())
    assert len(out0) == 1
    assert len(out1) == 1
    assert out0[0] == payload
    assert out1[0] == payload
    assert out0[0] is not payload
    assert out1[0] is not payload
    assert out0[0] is not out1[0]


def test_broadcast_stage_honors_stop_event() -> None:
    stage = BroadcastStage[int](
        config=BaseBroadcastStageConfig(
            n_outputs=2,
            out_maxsize=8,
            out_timeout=0.01,
        )
    )
    input_q: BufferQ[int] = BufferQ(maxsize=8, default_timeout=0.01)
    input_q.put(1)
    input_q.put(2)
    input_q.close()
    stop = threading.Event()
    stop.set()

    stage.run(input_q, stop=stop)

    assert list(stage.output(0).iter_until_closed()) == []
    assert list(stage.output(1).iter_until_closed()) == []


def test_split_stage_routes_items_and_closes_outputs() -> None:
    stage = _ParitySplit()
    input_q: BufferQ[int] = BufferQ(maxsize=16, default_timeout=0.01)
    for value in [-1, 0, 1, 2, 3, 4]:
        input_q.put(value)
    input_q.close()

    stage.run(input_q)

    assert stage.output_a.closed is True
    assert stage.output_b.closed is True
    assert list(stage.output_a.iter_until_closed()) == ["even:0", "even:2", "even:4"]
    assert list(stage.output_b.iter_until_closed()) == [1, 3]


def test_broadcast_stage_threaded_fans_out_to_all_outputs() -> None:
    stage = BroadcastStage[int](
        config=BaseBroadcastStageConfig(
            n_outputs=2,
            out_maxsize=8,
            out_timeout=0.01,
        )
    )
    input_q: BufferQ[int] = BufferQ(maxsize=8, default_timeout=0.01)
    for value in [3, 4, 5]:
        input_q.put(value)
    input_q.close()

    task = stage.threaded(input_q, daemon=False, join_timeout=1.0)
    task.start()
    task.join()

    assert stage.output(0).closed is True
    assert stage.output(1).closed is True
    assert list(stage.output(0).iter_until_closed()) == [3, 4, 5]
    assert list(stage.output(1).iter_until_closed()) == [3, 4, 5]


def test_split_stage_threaded_routes_and_closes_outputs() -> None:
    stage = _ParitySplit()
    input_q: BufferQ[int] = BufferQ(maxsize=16, default_timeout=0.01)
    for value in [0, 1, 2, 3]:
        input_q.put(value)
    input_q.close()

    task = stage.threaded(input_q, daemon=False, join_timeout=1.0)
    task.start()
    task.join()

    assert stage.output_a.closed is True
    assert stage.output_b.closed is True
    assert list(stage.output_a.iter_until_closed()) == ["even:0", "even:2"]
    assert list(stage.output_b.iter_until_closed()) == [1, 3]
