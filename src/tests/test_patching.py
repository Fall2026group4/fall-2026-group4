import numpy as np
import torch
from transformers import AutoTokenizer

from src.evaluation import patching as pt


def test_kl_zero_and_positive():
    lp = torch.log_softmax(torch.randn(4, 10), -1)
    lq = torch.log_softmax(torch.randn(4, 10), -1)
    assert torch.allclose(pt.kl(lp, lp), torch.zeros(4, dtype=torch.float64))
    assert (pt.kl(lp, lq) > 0).all()


def test_position_groups_partition_prompt_after_slot():
    tok = AutoTokenizer.from_pretrained('gpt2')
    instr = "Pretend you're a truthful person making statements about the world."
    p = pt.template_positions(tok, instr, 8)
    final = p['asst_end'] + 3
    allpos = sum((pt.group_positions(g, p['slot'], p['asst_start'], p['asst_end'], final)
                  for g in pt.POSITION_GROUPS), [])
    assert sorted(allpos) == list(range(8, final + 1))          # every position from the slot to the final token once
    assert tok.decode([tok.encode(' ' + 'truthful')[0]]) == ' truthful'


def test_holm_and_stat_summary():
    assert np.allclose(pt.holm([0.01, 0.02, 0.5]), [0.03, 0.04, 0.5])
    s = pt.stat_summary(np.r_[np.ones(10), np.nan], np.r_[np.arange(10), 10], 200, np.random.default_rng(0))
    assert s['mean'] == 1 and s['n_statements'] == 10


def test_random_like_matches_norm():
    d = torch.randn(3, 5, 8)
    d[0, 0] = 0
    r = pt.random_like(d, torch.Generator().manual_seed(0))
    assert torch.allclose(r.norm(dim=-1), d.norm(dim=-1), atol=1e-5)
