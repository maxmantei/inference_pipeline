from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import pytest

from inference_pipeline.configs import BaseConsumerConfig, BaseProcessorConfig, BaseProducerConfig
from inference_pipeline.pipeline import (
    PipelineBuilder,
    PipelineManager,
    PipelineNotStarted,
    PipelineOutcome,
    PipelinePhase,
    PipelineRunActive,
    PipelineRunCompleted,
    PipelineRunNotFound,
    PipelineStopMode,
    ResourceLimitExceeded,
    RunInfo,
)
from inference_pipeline.stages import ConsumerStage, ProcessorStage, ProducerStage


class _CollectStr(ConsumerStage[str, BaseConsumerConfig]):
    def __init__(self, *, config: BaseConsumerConfig) -> None:
        super().__init__(config=config)
        self.items: list[str] = []
        self.fail_on = config.name

    def consume(self, item: str) -> None:
        if item == self.fail_on:
            raise RuntimeError(f"failed on {item}")
        self.items.append(item)


class _RangeSource(ProducerStage[int, BaseProducerConfig]):
    def produce(self, stop: threading.Event | None = None) -> Iterator[int]:
        for value in [1, 2, 3]:
            if stop is not None and stop.is_set():
                return
            yield value


class _RenderingProcessor(ProcessorStage[int, str, BaseProcessorConfig]):
    def process(self, item: int) -> str | None:
        return f"value:{item}"


def _source_config() -> BaseProducerConfig:
    return BaseProducerConfig(out_maxsize=8, out_timeout=0.01)


def _processor_config() -> BaseProcessorConfig:
    return BaseProcessorConfig(out_maxsize=8, out_timeout=0.01)


def _runtime_spec(*, fail_on: str | None = None):
    builder = PipelineBuilder("runtime")
    (
        builder.source("frames", _RangeSource, config=_source_config())
        .then("render", _RenderingProcessor, config=_processor_config())
        .sink("writer", _CollectStr, config=BaseConsumerConfig(name=fail_on))
    )
    return builder.freeze()


def test_manager_starts_empty() -> None:
    manager = PipelineManager()

    assert manager.created == {}
    assert manager.running == {}
    assert manager.completed == {}


def test_manager_create_run_registers_created_pipeline() -> None:
    manager = PipelineManager()

    run = manager.create_run(_runtime_spec())

    assert run.phase is PipelinePhase.CREATED
    assert set(manager.created) == {"runtime-run-1"}


def test_manager_start_run_moves_to_running_view() -> None:
    manager = PipelineManager()
    run = manager.create_run(_runtime_spec())

    started = manager.start_run(run.run_id)

    assert started.phase is PipelinePhase.RUNNING
    assert set(manager.running) == {run.run_id}


def test_manager_run_completes_pipeline() -> None:
    manager = PipelineManager()

    run = manager.run(_runtime_spec())

    assert run.phase is PipelinePhase.TERMINATED
    assert run.outcome is PipelineOutcome.SUCCEEDED
    assert set(manager.completed) == {run.run_id}


def test_manager_wait_is_repeatable_for_terminated_runs() -> None:
    manager = PipelineManager()
    run = manager.run(_runtime_spec())

    again = manager.wait(run.run_id)

    assert again.run_id == run.run_id
    assert again.outcome is PipelineOutcome.SUCCEEDED


def test_manager_request_stop_then_wait_transitions_to_stopped() -> None:
    manager = PipelineManager()
    run = manager.start(_runtime_spec())

    manager.request_stop(run.run_id)
    stopped = manager.wait(run.run_id)

    assert stopped.phase is PipelinePhase.TERMINATED
    assert stopped.outcome is PipelineOutcome.STOPPED
    assert stopped.stop_mode is PipelineStopMode.IMMEDIATE


def test_manager_request_drain_then_wait_transitions_to_succeeded() -> None:
    manager = PipelineManager()
    run = manager.start(_runtime_spec())

    draining = manager.request_drain(run.run_id)
    done = manager.wait(run.run_id)

    assert draining.stop_mode is PipelineStopMode.DRAIN
    assert done.phase is PipelinePhase.TERMINATED
    assert done.outcome is PipelineOutcome.SUCCEEDED


def test_manager_drain_requests_and_waits() -> None:
    manager = PipelineManager()
    run = manager.start(_runtime_spec())

    done = manager.drain(run.run_id)

    assert done.phase is PipelinePhase.TERMINATED
    assert done.stop_mode is PipelineStopMode.DRAIN
    assert done.outcome is PipelineOutcome.SUCCEEDED


def test_manager_request_stop_rejects_not_started_run() -> None:
    manager = PipelineManager()
    run = manager.create_run(_runtime_spec())

    with pytest.raises(PipelineNotStarted):
        manager.request_stop(run.run_id)


def test_manager_request_stop_rejects_completed_run() -> None:
    manager = PipelineManager()
    run = manager.run(_runtime_spec())

    with pytest.raises(PipelineRunCompleted):
        manager.request_stop(run.run_id)


def test_manager_cancel_requests_stop_and_waits() -> None:
    manager = PipelineManager()
    run = manager.start(_runtime_spec())

    cancelled = manager.cancel(run.run_id)

    assert cancelled.phase is PipelinePhase.TERMINATED
    assert cancelled.outcome is PipelineOutcome.STOPPED
    assert run.run_id in manager.completed


def test_manager_discard_rejects_active_run() -> None:
    manager = PipelineManager()
    run = manager.start(_runtime_spec())

    with pytest.raises(PipelineRunActive):
        manager.discard(run.run_id)


def test_manager_discard_removes_terminated_run() -> None:
    manager = PipelineManager()
    run = manager.run(_runtime_spec())

    manager.discard(run.run_id)

    with pytest.raises(PipelineRunNotFound):
        manager.pipeline(run.run_id)


def test_manager_resource_limit_uses_active_runs() -> None:
    manager = PipelineManager(max_active_runs=1)
    run = manager.start(_runtime_spec())

    with pytest.raises(ResourceLimitExceeded):
        manager.create_run(_runtime_spec())

    manager.request_stop(run.run_id)
    manager.wait(run.run_id)

    next_run = manager.create_run(_runtime_spec())
    assert next_run.run_id == "runtime-run-2"


def test_manager_run_info_reports_phase_outcome_and_flags() -> None:
    manager = PipelineManager()
    created = manager.create_run(_runtime_spec())

    info_created = manager.run_info(created.run_id)
    assert isinstance(info_created, RunInfo)
    assert info_created.phase is PipelinePhase.CREATED
    assert info_created.outcome is None
    assert info_created.stop_mode is PipelineStopMode.NONE
    assert info_created.done is False

    manager.start_run(created.run_id)
    manager.request_stop(created.run_id)
    manager.wait(created.run_id)

    info_done = manager.run_info(created.run_id)
    assert info_done.phase is PipelinePhase.TERMINATED
    assert info_done.outcome is PipelineOutcome.STOPPED
    assert info_done.stop_mode is PipelineStopMode.IMMEDIATE
    assert info_done.done is True
    assert info_done.started_at is not None
    assert info_done.finished_at is not None
    assert info_done.stop_mode_requested_at is not None


def test_manager_runs_by_phase_and_outcome() -> None:
    manager = PipelineManager()
    manager.create_run(_runtime_spec())
    manager.run(_runtime_spec())

    by_phase = manager.runs_by_phase()
    by_outcome = manager.runs_by_outcome()

    assert by_phase[PipelinePhase.CREATED] == 1
    assert by_phase[PipelinePhase.TERMINATED] == 1
    assert by_outcome[PipelineOutcome.SUCCEEDED] == 1


def test_manager_health_reports_new_keys() -> None:
    manager = PipelineManager()
    manager.run(_runtime_spec())

    health = manager.health()

    assert health["created_count"] == 0
    assert health["running_count"] == 0
    assert health["completed_count"] == 1
    assert health["active_count"] == 0
    assert "runs_by_phase" in health
    assert "runs_by_outcome" in health


def test_manager_ttl_removes_terminated_runs() -> None:
    manager = PipelineManager(completed_run_ttl=0.1)
    run = manager.run(_runtime_spec())
    assert run.run_id in manager.completed

    time.sleep(0.3)

    with pytest.raises(PipelineRunNotFound):
        manager.pipeline(run.run_id)


def test_manager_unknown_run_id_operations_fail() -> None:
    manager = PipelineManager()

    with pytest.raises(PipelineRunNotFound):
        manager.pipeline("missing")
    with pytest.raises(PipelineRunNotFound):
        manager.wait("missing")
    with pytest.raises(PipelineRunNotFound):
        manager.request_stop("missing")
