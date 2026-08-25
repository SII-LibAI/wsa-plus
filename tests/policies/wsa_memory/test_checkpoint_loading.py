from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn


def _load_checkpoint_module():
    path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "lerobot"
        / "policies"
        / "WSA_Memory"
        / "checkpoint_loading.py"
    )
    spec = importlib.util.spec_from_file_location(
        "wsa_memory_checkpoint_loading_under_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checkpoint_loading = _load_checkpoint_module()


class _TiedEmbeddingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.lm_head.weight = self.embed_tokens.weight


def test_shared_embedding_alias_is_not_reported_missing(tmp_path: Path) -> None:
    from safetensors.torch import load_file, save_model

    source = _TiedEmbeddingModel()
    with torch.no_grad():
        source.embed_tokens.weight.copy_(
            torch.arange(32, dtype=torch.float32).reshape(8, 4)
        )
    checkpoint_path = tmp_path / "model.safetensors"
    save_model(source, checkpoint_path)

    # safetensors stores only one name for the tied storage.
    assert len(load_file(checkpoint_path)) == 1

    target = _TiedEmbeddingModel()
    with torch.no_grad():
        target.embed_tokens.weight.zero_()
    loaded, missing, unexpected = (
        checkpoint_loading.load_shared_safetensors_checkpoint(
            target,
            checkpoint_path,
        )
    )

    assert missing == []
    assert unexpected == []
    assert loaded == {"embed_tokens.weight", "lm_head.weight"}
    torch.testing.assert_close(
        target.embed_tokens.weight,
        source.embed_tokens.weight,
    )
    assert target.embed_tokens.weight.data_ptr() == target.lm_head.weight.data_ptr()
