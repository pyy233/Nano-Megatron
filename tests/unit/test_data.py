import pytest

torch = pytest.importorskip("torch")

from nano_megatron.data import FixedLengthTokenDataset, RandomTokenDataset  # noqa: E402


def test_fixed_length_dataset_builds_shifted_labels() -> None:
    dataset = FixedLengthTokenDataset(torch.arange(15), sequence_length=4)
    assert len(dataset) == 3
    sample = dataset[1]
    torch.testing.assert_close(sample["input_ids"], torch.tensor([5, 6, 7, 8]))
    torch.testing.assert_close(sample["labels"], torch.tensor([6, 7, 8, 9]))


def test_random_dataset_is_deterministic_per_index() -> None:
    dataset = RandomTokenDataset(num_samples=2, sequence_length=8, vocab_size=32, seed=9)
    torch.testing.assert_close(dataset[1]["input_ids"], dataset[1]["input_ids"])
    assert not torch.equal(dataset[0]["input_ids"], dataset[1]["input_ids"])
