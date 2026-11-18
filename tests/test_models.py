"""TASK-004: FR-002, AC-004/005 và hợp đồng build_model trong docs/05_model.md."""
from copy import deepcopy
from io import BytesIO
from pathlib import Path

import pytest
import torch
from torch import nn

from src.models import build_model
from src.models.vit import DropPath
from src.utils import count_params, load_config

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "vit_small.yaml"


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    # Tensor nhỏ trong unit test không cần tạo nhiều luồng CPU.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def cfg():
    return load_config(CONFIG)


@pytest.fixture
def small_cfg(cfg):
    cfg["data"].update(img_size=32, num_classes=7)
    cfg["model"].update(embed_dim=48, depth=2, num_heads=3)
    return cfg


@pytest.fixture(scope="module")
def default_run():
    torch.manual_seed(42)
    model = build_model(load_config(CONFIG)).eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        logits = model(x)
        explicit_logits, attns = model(x, return_attn=True)
    return model, logits, explicit_logits, attns


def test_default_forward_shape_and_parameter_count(default_run):
    model, logits, _, _ = default_run
    assert logits.shape == (2, 200)
    assert torch.isfinite(logits).all()
    assert count_params(model) == 6_436_296


def test_default_attention_contract_and_matching_logits(default_run):
    _, logits, explicit_logits, attns = default_run
    assert isinstance(attns, list) and len(attns) == 8
    for attn in attns:
        assert attn.shape == (2, 8, 65, 65)
        assert torch.isfinite(attn).all() and (attn >= 0).all()
        torch.testing.assert_close(attn.sum(-1), torch.ones(2, 8, 65), rtol=0, atol=1e-4)
    torch.testing.assert_close(logits, explicit_logits, rtol=0, atol=1e-4)


@pytest.mark.parametrize("img_size,patch_size,tokens", [(32, 4, 65), (64, 16, 17)])
def test_factory_uses_config_for_grid_classes_and_encoder(small_cfg, img_size, patch_size, tokens):
    small_cfg["data"]["img_size"] = img_size
    small_cfg["model"]["patch_size"] = patch_size
    original = deepcopy(small_cfg)
    model = build_model(small_cfg).eval()
    with torch.no_grad():
        logits, attns = model(torch.randn(2, 3, img_size, img_size), return_attn=True)
    assert small_cfg == original
    assert logits.shape == (2, 7)
    assert len(attns) == 2 and all(a.shape == (2, 3, tokens, tokens) for a in attns)


@pytest.mark.parametrize("name", ["unknown", "vit_pretrained"])
def test_factory_rejects_unknown_model_with_valid_names(cfg, name):
    cfg["model"]["name"] = name
    with pytest.raises(ValueError, match="vit"):
        build_model(cfg)


@pytest.mark.parametrize("section,key,value", [
    ("data", "img_size", 63),
    ("model", "patch_size", 0),
    ("model", "embed_dim", 250),
    ("model", "num_heads", 0),
    ("model", "depth", 0),
    ("model", "mlp_ratio", 0),
    ("model", "dropout", -0.1),
    ("model", "drop_path_rate", 1.0),
])
def test_invalid_config_reports_relevant_field(cfg, section, key, value):
    cfg[section][key] = value
    with pytest.raises(ValueError, match=key):
        build_model(cfg)


@pytest.mark.parametrize("shape", [(2, 1, 32, 32), (2, 3, 16, 64), (3, 32, 32)])
def test_forward_rejects_input_incompatible_with_patch_grid(small_cfg, shape):
    model = build_model(small_cfg)
    with pytest.raises(ValueError, match="input"):
        model(torch.randn(shape))


@pytest.mark.parametrize("return_attn", [False, True])
def test_training_backward_and_optimizer_update(small_cfg, return_attn):
    torch.manual_seed(42)
    model = build_model(small_cfg).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.head.weight.detach().clone()
    output = model(torch.randn(2, 3, 32, 32), return_attn=return_attn)
    logits = output[0] if return_attn else output
    loss = nn.CrossEntropyLoss()(logits, torch.tensor([1, 5]))
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    optimizer.step()
    assert not torch.equal(before, model.head.weight)


def test_eval_disables_dropout_and_drop_path_in_both_attention_branches(small_cfg):
    small_cfg["model"].update(dropout=0.4, drop_path_rate=0.5)
    model = build_model(small_cfg).eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        first = model(x)
        second = model(x)
        explicit, _ = model(x, return_attn=True)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first, explicit, rtol=0, atol=1e-4)


def test_training_returns_attention_before_dropout(small_cfg):
    torch.manual_seed(42)
    small_cfg["model"]["dropout"] = 0.5
    model = build_model(small_cfg).train()
    x = torch.randn(2, 3, 32, 32)
    logits, attns = model(x, return_attn=True)
    assert all(a.requires_grad for a in attns)
    for attn in attns:
        assert (attn > 0).all()  # dropout 0,5 không được xóa phần tử trong weights trả về
        torch.testing.assert_close(attn.sum(-1), torch.ones(2, 3, 17), rtol=0, atol=1e-4)
    assert not torch.equal(logits, model(x, return_attn=True)[0])


def test_drop_path_drops_whole_samples_and_preserves_expected_scale():
    torch.manual_seed(42)
    x = torch.ones(128, 7, 5)
    drop = DropPath(0.5).train()
    result = drop(x)
    assert set(result.unique().tolist()) == {0.0, 2.0}
    assert torch.equal(result, result[:, :1, :1].expand_as(result))
    torch.testing.assert_close(drop.eval()(x), x, rtol=0, atol=0)
    torch.testing.assert_close(DropPath(0).train()(x), x, rtol=0, atol=0)


def test_checkpoint_round_trip_with_weights_only(small_cfg):
    model = build_model(small_cfg).eval()
    x = torch.randn(2, 3, 32, 32)
    buffer = BytesIO()
    torch.save({"model": model.state_dict(), "config": small_cfg}, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)
    restored = build_model(checkpoint["config"]).eval()
    restored.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)
def test_resnet_small_matches_vit_parameter_count():
    cnn_cfg = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "cnn_baseline.yaml"
    )

    cnn = build_model(cnn_cfg)
    cnn_params = count_params(cnn)

    vit_params = 6_436_296
    relative_error = abs(cnn_params - vit_params) / vit_params

    assert relative_error <= 0.05