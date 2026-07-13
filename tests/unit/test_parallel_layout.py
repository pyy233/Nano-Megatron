import pytest

from nano_megatron.parallel import TensorLayout


def test_tensor_layout_accepts_combined_sequence_and_hidden_sharding() -> None:
    layout = TensorLayout(
        cp_sharded=True,
        sp_sharded=True,
        tp_hidden_sharded=True,
    )

    assert layout.sequence_dim == 1


@pytest.mark.parametrize("field", ["cp_sharded", "sp_sharded", "tp_hidden_sharded"])
def test_tensor_layout_rejects_non_boolean_shard_flags(field: str) -> None:
    values = {
        "cp_sharded": False,
        "sp_sharded": False,
        "tp_hidden_sharded": False,
    }
    values[field] = 1

    with pytest.raises(TypeError, match=field):
        TensorLayout(**values)
