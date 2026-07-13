from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn

from nano_megatron.config import (
    DataParallelConfig,
    DistributedConfig,
    GPTConfig,
    OffloadConfig,
    OptimizerConfig,
    ParallelConfig,
)
from nano_megatron.data_parallel import build_data_parallel_strategy
from nano_megatron.distributed import DistributedRuntime
from nano_megatron.models.gpt import GPTModel, GPTModelBuilder
from nano_megatron.nn.kernels import TorchKernelBackend
from nano_megatron.parallel import ParallelContext
from nano_megatron.pipeline_parallel import (
    GPipeSchedule,
    InterleavedOneForwardOneBackwardSchedule,
    OneForwardOneBackwardSchedule,
    P2PCommunicator,
)


@dataclass(frozen=True)
class _LocalGroup:
    rank: int = 0
    size: int = 1
    process_group: object | None = None


@dataclass(frozen=True)
class _SingleParallel:
    tp: _LocalGroup = _LocalGroup()
    pp: _LocalGroup = _LocalGroup()
    cp: _LocalGroup = _LocalGroup()
    sequence_parallel: bool = False


def _copy_reference_partition(reference: GPTModel, stage: GPTModel) -> None:
    with torch.no_grad():
        if stage.embedding is not None:
            assert reference.embedding is not None
            stage.embedding.load_state_dict(reference.embedding.state_dict())
        for local_layer, global_layer_index in zip(
            stage.layers,
            range(stage.layer_start, stage.layer_end),
            strict=True,
        ):
            local_layer.load_state_dict(reference.layers[global_layer_index].state_dict())
        if stage.final_norm is not None:
            assert reference.final_norm is not None
            stage.final_norm.load_state_dict(reference.final_norm.state_dict())
        if stage.lm_head is not None:
            # Deliberately make the last-stage head wrong. The schedule's
            # pre-P2P lifecycle hook must broadcast the first-stage embedding.
            stage.lm_head.weight.fill_(123.0)


def _assert_module_gradients(actual: nn.Module, expected: nn.Module) -> None:
    actual_parameters = dict(actual.named_parameters())
    expected_parameters = dict(expected.named_parameters())
    assert actual_parameters.keys() == expected_parameters.keys()
    for name, actual_parameter in actual_parameters.items():
        expected_parameter = expected_parameters[name]
        assert actual_parameter.grad is not None, name
        assert expected_parameter.grad is not None, name
        torch.testing.assert_close(
            actual_parameter.grad,
            expected_parameter.grad,
            atol=2.0e-5,
            rtol=2.0e-4,
        )


def _assert_module_parameters(actual: nn.Module, expected: nn.Module) -> None:
    actual_parameters = dict(actual.named_parameters())
    expected_parameters = dict(expected.named_parameters())
    assert actual_parameters.keys() == expected_parameters.keys()
    for name, actual_parameter in actual_parameters.items():
        full_tensor = getattr(actual_parameter, "full_tensor", None)
        actual_value = full_tensor() if callable(full_tensor) else actual_parameter
        torch.testing.assert_close(
            actual_value,
            expected_parameters[name],
            atol=2.0e-5,
            rtol=2.0e-4,
        )


def _pipeline_worker(rank: int, rendezvous: str, schedule_name: str) -> None:
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank))
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    runtime = DistributedRuntime(
        DistributedConfig(
            backend="gloo",
            device="cpu",
            init_method=f"file://{rendezvous}",
        )
    ).initialize()
    parallel = ParallelContext.create(
        runtime,
        ParallelConfig(pipeline=2, data=1),
    )
    try:
        config = GPTConfig(
            layers=2,
            hidden_size=8,
            ffn_hidden_size=16,
            heads=2,
            kv_heads=2,
            seq_length=4,
            vocab_size=16,
            dropout=0.0,
            tie_embeddings=True,
            bias=False,
        )
        kernels = TorchKernelBackend()
        torch.manual_seed(2027)
        reference = GPTModel(
            config,
            parallel=_SingleParallel(),
            kernels=kernels,
        )
        built = GPTModelBuilder().build_stage(config, parallel, kernels)
        stage = built.model
        _copy_reference_partition(reference, stage.model)
        assert stage.tied_embeddings is not None
        assert not stage.tied_embeddings.weight_is_synchronized

        microbatches = [
            {
                "input_ids": torch.tensor([[0, 2, 5, 9]]),
                "labels": torch.tensor([[2, 5, 9, 1]]),
            },
            {
                "input_ids": torch.tensor([[3, 7, 11, 15]]),
                "labels": torch.tensor([[7, 11, 15, 4]]),
            },
            {
                "input_ids": torch.tensor([[6, 1, 12, 8]]),
                "labels": torch.tensor([[1, 12, 8, 10]]),
            },
        ]

        reference_optimizer = torch.optim.AdamW(
            reference.parameters(),
            lr=1.0e-2,
            weight_decay=0.0,
        )
        stage_optimizer = torch.optim.AdamW(
            stage.parameters(),
            lr=1.0e-2,
            weight_decay=0.0,
        )
        reference_loss = torch.zeros(())
        for batch in microbatches:
            output = reference(batch["input_ids"], labels=batch["labels"])
            assert output.loss is not None
            scaled_loss = output.loss / len(microbatches)
            reference_loss = reference_loss + scaled_loss.detach()
            scaled_loss.backward()

        overlap_dynamic = schedule_name.endswith("-overlap-dynamic")
        communicator = P2PCommunicator(
            parallel,
            activation_shape=(
                None if overlap_dynamic else (1, config.seq_length, config.hidden_size)
            ),
            activation_dtype=torch.float32,
            device="cpu",
            dynamic_shapes=overlap_dynamic,
        )
        if schedule_name.startswith("gpipe"):
            schedule = GPipeSchedule(
                parallel,
                communicator,
                overlap_p2p=overlap_dynamic,
            )
        else:
            schedule = OneForwardOneBackwardSchedule(
                parallel,
                communicator,
                overlap_p2p=overlap_dynamic,
            )
        result = schedule.forward_backward(
            stage=stage,
            microbatches=microbatches,
        )

        assert stage.tied_embeddings.weight_is_synchronized
        assert reference.embedding is not None
        if parallel.is_pipeline_last_stage():
            torch.testing.assert_close(
                sum(result.losses),
                reference_loss,
                atol=2.0e-5,
                rtol=2.0e-4,
            )
            assert stage.model.lm_head is not None
            torch.testing.assert_close(
                stage.model.lm_head.weight,
                reference.embedding.weight,
            )
            assert stage.model.lm_head.weight.grad is not None
            assert reference.embedding.weight.grad is not None
            torch.testing.assert_close(
                stage.model.lm_head.weight.grad,
                reference.embedding.weight.grad,
                atol=2.0e-5,
                rtol=2.0e-4,
            )
        else:
            assert result.losses == ()
            assert stage.model.embedding is not None
            assert stage.model.embedding.weight.grad is not None
            assert reference.embedding.weight.grad is not None
            torch.testing.assert_close(
                stage.model.embedding.weight.grad,
                reference.embedding.weight.grad,
                atol=2.0e-5,
                rtol=2.0e-4,
            )

        for local_layer, global_layer_index in zip(
            stage.model.layers,
            range(stage.model.layer_start, stage.model.layer_end),
            strict=True,
        ):
            _assert_module_gradients(local_layer, reference.layers[global_layer_index])
        if stage.model.final_norm is not None:
            assert reference.final_norm is not None
            _assert_module_gradients(stage.model.final_norm, reference.final_norm)

        reference_optimizer.step()
        stage_optimizer.step()
        if stage.model.embedding is not None:
            assert reference.embedding is not None
            torch.testing.assert_close(
                stage.model.embedding.weight,
                reference.embedding.weight,
                atol=2.0e-5,
                rtol=2.0e-4,
            )
        if stage.model.lm_head is not None:
            assert reference.embedding is not None
            torch.testing.assert_close(
                stage.model.lm_head.weight,
                reference.embedding.weight,
                atol=2.0e-5,
                rtol=2.0e-4,
            )
        for local_layer, global_layer_index in zip(
            stage.model.layers,
            range(stage.model.layer_start, stage.model.layer_end),
            strict=True,
        ):
            _assert_module_parameters(local_layer, reference.layers[global_layer_index])
        if stage.model.final_norm is not None:
            assert reference.final_norm is not None
            _assert_module_parameters(stage.model.final_norm, reference.final_norm)
    finally:
        parallel.close()
        runtime.close()


def _interleaved_gpt_worker(rank: int, rendezvous: str) -> None:
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank))
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    runtime = DistributedRuntime(
        DistributedConfig(
            backend="gloo",
            device="cpu",
            init_method=f"file://{rendezvous}",
        )
    ).initialize()
    parallel = ParallelContext.create(
        runtime,
        ParallelConfig(pipeline=2, data=1),
    )
    try:
        config = GPTConfig(
            layers=4,
            hidden_size=8,
            ffn_hidden_size=16,
            heads=2,
            kv_heads=2,
            seq_length=4,
            vocab_size=16,
            dropout=0.0,
            tie_embeddings=True,
            bias=False,
        )
        kernels = TorchKernelBackend()
        torch.manual_seed(4021)
        reference = GPTModel(config, parallel=_SingleParallel(), kernels=kernels)
        built = GPTModelBuilder().build_pipeline(
            config,
            parallel,
            kernels,
            virtual_stages_per_rank=2,
        )
        pipeline = built.model
        for chunk in pipeline.chunks:
            _copy_reference_partition(reference, chunk.model)

        microbatches = [
            {
                "input_ids": torch.tensor([[0, 2, 5, 9]]),
                "labels": torch.tensor([[2, 5, 9, 1]]),
            },
            {
                "input_ids": torch.tensor([[3, 7, 11, 15]]),
                "labels": torch.tensor([[7, 11, 15, 4]]),
            },
        ]
        reference_losses = []
        for batch in microbatches:
            output = reference(batch["input_ids"], labels=batch["labels"])
            assert output.loss is not None
            reference_losses.append(output.loss.detach())
            (output.loss / len(microbatches)).backward()

        communicator = P2PCommunicator(
            parallel,
            activation_shape=None,
            activation_dtype=torch.float32,
            device="cpu",
            dynamic_shapes=True,
        )
        result = InterleavedOneForwardOneBackwardSchedule(
            parallel,
            built.layout,
            communicator,
            overlap_p2p=True,
        ).forward_backward(stage=pipeline, microbatches=microbatches)

        if parallel.is_pipeline_last_stage():
            torch.testing.assert_close(
                sum(result.losses),
                sum(reference_losses) / len(reference_losses),
                atol=3.0e-5,
                rtol=3.0e-4,
            )
        else:
            assert result.losses == ()
        for chunk in pipeline.chunks:
            for local_layer, global_layer_index in zip(
                chunk.model.layers,
                range(chunk.model.layer_start, chunk.model.layer_end),
                strict=True,
            ):
                _assert_module_gradients(
                    local_layer,
                    reference.layers[global_layer_index],
                )
        endpoint = pipeline.chunks[0] if rank == 0 else pipeline.chunks[-1]
        tied_weight = (
            endpoint.model.embedding.weight
            if endpoint.model.embedding is not None
            else endpoint.model.lm_head.weight
        )
        assert tied_weight.grad is not None
        assert reference.embedding is not None
        torch.testing.assert_close(
            tied_weight.grad,
            reference.embedding.weight.grad,
            atol=3.0e-5,
            rtol=3.0e-4,
        )
    finally:
        parallel.close()
        runtime.close()


def _pipeline_zero_worker(rank: int, rendezvous: str, zero_mode: str) -> None:
    os.environ.update(RANK=str(rank), WORLD_SIZE="4", LOCAL_RANK=str(rank))
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    runtime = DistributedRuntime(
        DistributedConfig(
            backend="gloo",
            device="cpu",
            init_method=f"file://{rendezvous}",
        )
    ).initialize()
    parallel = ParallelContext.create(
        runtime,
        ParallelConfig(pipeline=2, data=2),
    )
    try:
        config = GPTConfig(
            layers=2,
            hidden_size=8,
            ffn_hidden_size=16,
            heads=2,
            kv_heads=2,
            seq_length=4,
            vocab_size=16,
            dropout=0.0,
            tie_embeddings=True,
            bias=False,
        )
        optimizer_config = OptimizerConfig(
            lr=1.0e-2,
            betas=(0.9, 0.95),
            eps=1.0e-8,
            weight_decay=0.0,
            clip_grad_norm=None,
        )
        kernels = TorchKernelBackend()
        torch.manual_seed(3031)
        reference = GPTModel(
            config,
            parallel=_SingleParallel(),
            kernels=kernels,
        )
        built = GPTModelBuilder().build_stage(config, parallel, kernels)
        stage = built.model
        _copy_reference_partition(reference, stage.model)

        # Trainer performs this lifecycle hook before data_parallel.setup so
        # ZeRO's master parameter shard is initialized from the tied value.
        stage.synchronize_tied_embedding_weights()
        assert stage.tied_embeddings is not None
        assert stage.tied_embeddings.weight_is_synchronized
        if stage.model.lm_head is not None:
            assert reference.embedding is not None
            torch.testing.assert_close(
                stage.model.lm_head.weight,
                reference.embedding.weight,
            )

        strategy = build_data_parallel_strategy(
            DataParallelConfig(
                mode=zero_mode,
                bucket_bytes=512,
                overlap_grad_reduce=True,
            ),
            OffloadConfig(),
            parallel,
            built.parameter_domains,
        )
        stage = strategy.setup(stage, optimizer_config, built.parameter_domains)
        reference_optimizer = torch.optim.AdamW(
            reference.parameters(),
            lr=optimizer_config.lr,
            betas=optimizer_config.betas,
            eps=optimizer_config.eps,
            weight_decay=optimizer_config.weight_decay,
        )
        microbatches = [
            {
                "input_ids": torch.tensor([[0, 4, 8, 12]]),
                "labels": torch.tensor([[4, 8, 12, 1]]),
            },
            {
                "input_ids": torch.tensor([[3, 6, 10, 14]]),
                "labels": torch.tensor([[6, 10, 14, 2]]),
            },
        ]
        reference_loss = torch.zeros(())
        for batch in microbatches:
            output = reference(batch["input_ids"], labels=batch["labels"])
            assert output.loss is not None
            scaled_loss = output.loss / len(microbatches)
            reference_loss = reference_loss + scaled_loss.detach()
            scaled_loss.backward()

        communicator = P2PCommunicator(
            parallel,
            activation_shape=None,
            activation_dtype=torch.float32,
            device="cpu",
            dynamic_shapes=True,
        )
        schedule = OneForwardOneBackwardSchedule(
            parallel,
            communicator,
            overlap_p2p=True,
        )
        strategy.zero_grad()
        result = schedule.forward_backward(
            stage=stage,
            microbatches=microbatches,
            data_parallel=strategy,
        )
        if parallel.is_pipeline_last_stage():
            torch.testing.assert_close(
                sum(result.losses),
                reference_loss,
                atol=2.0e-5,
                rtol=2.0e-4,
            )

        core_stage = stage.model
        assert isinstance(core_stage, GPTModel)
        tied_weight = (
            core_stage.embedding.weight
            if core_stage.embedding is not None
            else core_stage.lm_head.weight
            if core_stage.lm_head is not None
            else None
        )
        assert tied_weight is not None and tied_weight.grad is not None
        assert reference.embedding is not None
        assert reference.embedding.weight.grad is not None
        full_gradient = getattr(tied_weight.grad, "full_tensor", None)
        tied_gradient = full_gradient() if callable(full_gradient) else tied_weight.grad
        torch.testing.assert_close(
            tied_gradient,
            reference.embedding.weight.grad,
            atol=3.0e-5,
            rtol=3.0e-4,
        )

        # This ordering is the core PP x ZeRO invariant: endpoint gradients
        # are summed first, then reduced/sharded across dense DP replicas.
        strategy.finalize_gradients()
        strategy.optimizer_step()
        reference_optimizer.step()

        if core_stage.embedding is not None:
            assert reference.embedding is not None
            full_tensor = getattr(core_stage.embedding.weight, "full_tensor", None)
            embedding_weight = (
                full_tensor() if callable(full_tensor) else core_stage.embedding.weight
            )
            torch.testing.assert_close(
                embedding_weight,
                reference.embedding.weight,
                atol=3.0e-5,
                rtol=3.0e-4,
            )
        if core_stage.lm_head is not None:
            assert reference.embedding is not None
            full_tensor = getattr(core_stage.lm_head.weight, "full_tensor", None)
            lm_head_weight = full_tensor() if callable(full_tensor) else core_stage.lm_head.weight
            torch.testing.assert_close(
                lm_head_weight,
                reference.embedding.weight,
                atol=3.0e-5,
                rtol=3.0e-4,
            )
        for local_layer, global_layer_index in zip(
            core_stage.layers,
            range(core_stage.layer_start, core_stage.layer_end),
            strict=True,
        ):
            _assert_module_parameters(local_layer, reference.layers[global_layer_index])
        if core_stage.final_norm is not None:
            assert reference.final_norm is not None
            _assert_module_parameters(core_stage.final_norm, reference.final_norm)
    finally:
        parallel.close()
        runtime.close()


@pytest.mark.distributed
@pytest.mark.parametrize(
    "schedule_name",
    ["gpipe", "gpipe-overlap-dynamic", "1f1b", "1f1b-overlap-dynamic"],
)
def test_two_stage_gpt_matches_eager_with_tied_embeddings(
    tmp_path: Path,
    schedule_name: str,
) -> None:
    rendezvous = tmp_path / f"gpt-{schedule_name}.rendezvous"
    mp.spawn(
        _pipeline_worker,
        args=(str(rendezvous), schedule_name),
        nprocs=2,
        join=True,
    )


@pytest.mark.distributed
def test_interleaved_gpt_matches_eager_with_tied_embeddings(
    tmp_path: Path,
) -> None:
    rendezvous = tmp_path / "gpt-interleaved.rendezvous"
    mp.spawn(
        _interleaved_gpt_worker,
        args=(str(rendezvous),),
        nprocs=2,
        join=True,
    )


@pytest.mark.distributed
@pytest.mark.parametrize("zero_mode", ["zero1", "zero2", "zero3"])
def test_pp2_dp2_tied_embeddings_match_eager_after_zero_step(
    tmp_path: Path,
    zero_mode: str,
) -> None:
    rendezvous = tmp_path / f"gpt-pp2-dp2-{zero_mode}.rendezvous"
    mp.spawn(
        _pipeline_zero_worker,
        args=(str(rendezvous), zero_mode),
        nprocs=4,
        join=True,
    )
