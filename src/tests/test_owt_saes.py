"""Pure-logic tests for the multi-SAE-type config helpers (no model download)."""
import pandas as pd
import pytest

from src.owt.saes import (
    hook_name_for_layer,
    layers_for_type,
    restrict_sae_types,
    sae_id_for_layer,
    save_merged,
)

CONFIG = {
    "layers": list(range(12)),
    "sae_types": {
        "relu": {
            "enabled": True,
            "point_type": "resid_pre",
            "hook_template": "blocks.{layer}.hook_resid_pre",
            "sae_id_template": "blocks.{source_layer}.hook_resid_pre",
        },
        "topk": {
            "enabled": True,
            "point_type": "resid_post",
            "layers": list(range(1, 12)),
            "hook_template": "blocks.{layer}.hook_resid_pre",
            "sae_id_template": "blocks.{source_layer}.hook_resid_post",
        },
    },
}


def test_relu_covers_all_positions_and_uses_same_block():
    assert layers_for_type(CONFIG, "relu") == list(range(12))
    assert sae_id_for_layer(CONFIG, "relu", 5) == "blocks.5.hook_resid_pre"


def test_topk_is_filed_one_position_later_than_its_block():
    assert layers_for_type(CONFIG, "topk") == list(range(1, 12))
    # position 5 = output of block 4
    assert sae_id_for_layer(CONFIG, "topk", 5) == "blocks.4.hook_resid_post"
    assert hook_name_for_layer(CONFIG, "topk", 5) == "blocks.5.hook_resid_pre"


def test_restrict_sae_types():
    import copy

    cfg = copy.deepcopy(CONFIG)
    restrict_sae_types(cfg, ["topk"])
    assert not cfg["sae_types"]["relu"]["enabled"] and cfg["sae_types"]["topk"]["enabled"]
    with pytest.raises(ValueError):
        restrict_sae_types(cfg, ["nope"])


def test_save_merged_keeps_other_types(tmp_path):
    path = tmp_path / "t.csv"
    pd.DataFrame({"sae_type": ["relu", "relu"], "layer": [0, 1], "v": [1, 2]}).to_csv(path, index=False)
    save_merged(pd.DataFrame({"sae_type": ["topk"], "layer": [1], "v": [9]}), path)
    out = pd.read_csv(path)
    assert sorted(out["sae_type"]) == ["relu", "relu", "topk"]
    # rerunning topk replaces, not duplicates
    save_merged(pd.DataFrame({"sae_type": ["topk"], "layer": [1], "v": [10]}), path)
    out = pd.read_csv(path)
    assert (out["sae_type"] == "topk").sum() == 1 and out.loc[out.sae_type == "topk", "v"].iloc[0] == 10
