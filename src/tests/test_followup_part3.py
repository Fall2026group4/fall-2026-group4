"""FinalTokenTail (scripts/repe/15_sae_in_the_loop_causal.py) must equal TransformerLens's own hooked forward pass.

Uses small randomly initialised HookedTransformer models (no download). Skipped without transformer_lens.
Run from the repository root:  python -m pytest -q src/tests/test_followup_part3.py
"""
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
tl = pytest.importorskip('transformer_lens')
REPO = Path(__file__).resolve().parents[2]


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


loop = load('scripts/repe/15_sae_in_the_loop_causal.py', 'repe15_tail')


def make_model(normalization, fold):
    cfg = tl.HookedTransformerConfig(n_layers=5, d_model=48, n_ctx=32, d_head=12, n_heads=4, d_vocab=97,
                                     d_mlp=96, act_fn='gelu_new', normalization_type=normalization, seed=0)
    m = tl.HookedTransformer(cfg)
    if fold:                       # what from_pretrained does for GPT-2: LayerNorm folded, LNPre at run time
        m.process_weights_(fold_ln=True, center_writing_weights=True, center_unembed=True)
    return m.eval()


def hooked_logp(model, tokens, final, layer, delta):
    rows = torch.arange(len(tokens))

    def fn(resid, hook):
        resid[rows, final] = resid[rows, final] + delta
        return resid
    logits = model.run_with_hooks(tokens, fwd_hooks=[(f'blocks.{layer}.hook_resid_pre', fn)])
    return logits[rows, final].float().log_softmax(-1)


@pytest.mark.parametrize('normalization,fold', [('LN', False), ('LN', True)])
@pytest.mark.parametrize('layer', [0, 2, 4])
def test_final_token_tail_equals_hooked_forward(normalization, fold, layer):
    torch.manual_seed(1)
    model = make_model(normalization, fold)
    lengths = torch.tensor([7, 12, 5, 12, 9])
    tokens = torch.randint(1, 97, (5, 12))
    for i, L in enumerate(lengths):
        tokens[i, L:] = 0                                   # right padding, as in Runner
    final = lengths - 1
    tail = loop.FinalTokenTail(model, tokens, final, layer)
    deltas = torch.randn(3, 5, 48) * 2
    deltas[1] = 0                                           # zero delta = baseline
    got = tail.logp(deltas)
    for j in range(3):
        want = hooked_logp(model, tokens, final, layer, deltas[j])
        assert torch.allclose(got[j], want, atol=1e-4), float((got[j] - want).abs().max())
    one = tail.logp(deltas[0, 0][None])                     # one shared delta for every prompt
    want = hooked_logp(model, tokens, final, layer, deltas[0, 0])
    assert torch.allclose(one[0], want, atol=1e-4)


def test_runner_paths_agree_on_a_real_model():
    torch.manual_seed(2)
    model = make_model('LN', True)

    class Tok:
        eos_token_id = 0
        padding_side = 'right'                              # GPT-2's tokenizer pads on the right
    model.tokenizer = Tok()
    model.to_tokens = lambda p, prepend_bos=True, move_to_device=False: torch.tensor(
        [[0] * prepend_bos + [ord(c) % 96 + 1 for c in p]])
    run = loop.Runner(model, ['hello there', 'a much longer prompt here', 'x'], 'blocks.3.hook_resid_pre', 'cpu')
    assert run.full_forward_check() < 1e-4                  # stop_at_layer path == full forward
    run.build_tail(3)
    ref = run.run()
    ds = [torch.randn(48), torch.randn(3, 48), torch.zeros(48)]
    fast = run.kl_many(ref, ds)
    slow = torch.stack([loop.kl_rows(ref, run.run(d)) for d in ds]).numpy()
    assert abs(fast - slow).max() < 1e-4
