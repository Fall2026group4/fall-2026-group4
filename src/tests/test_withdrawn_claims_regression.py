"""Regression guard: withdrawn scientific claims must not reappear in the paper or curated docs.

Decisions D38 (AUDIT-SENS S2, S5) and D36 (H1-TEST) withdrew or superseded several statements. The older tests in
test_audit_sensitivity.py and test_h1_test_paper_claims.py match exact strings, including line breaks, so a reworded
or re-wrapped version of a withdrawn claim (for example after the manuscript restructure in PR #31) can slip past
them. This test strips LaTeX/Markdown markup, comments and whitespace differences before matching, and also scans
the curated plain-language documents.

A match is allowed only when the same sentence marks the claim as historical (e.g. "withdrawn", "superseded",
"invalid"), so a document may still report that a claim was withdrawn.

Ledgers such as RESEARCH_PROGRESS.md and RESEARCH_DECISIONS.md are not scanned: they record withdrawn claims on
purpose.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# Files that present conclusions to readers. Missing files are skipped, so the test works on every branch.
SCANNED_GLOBS = (
    'research_paper/Latex/sections/*.tex',
    'reports/Latex_report/main.tex',
    'docs/plain_summary.md',
    'docs/research_overview.md',
    'docs/results_summary.md',
    'docs/limitations_and_future_work.md',
    'presentation/gpt2_results_slides.md',
    'README.md',
)

# (phrase, reason). Phrases are compared with normalised text (see normalise()).
WITHDRAWN = (
    # D38 / AUDIT-SENS S5: 33-50 % of the GoT frozen-logistic drop at layers 6-11 is label-aligned SAE error.
    ('reflects the sensitivity of that reader to any perturbation', 'S5: frozen GoT collapse is not mostly generic perturbation'),
    ('the frozen truth loss is mostly reader fragility', 'S5: frozen GoT collapse is not mostly reader fragility'),
    ('is largely perturbation sensitivity', 'S5: withdrawn as worded'),
    ('is largely reproduced by random noise of the same norm', 'S5: withdrawn as worded'),
    ('so the probe was reacting to any disturbance', 'S5: plain-summary form of the withdrawn claim'),
    ('most of the frozen-probe collapse is therefore', 'S5: withdrawn as worded'),
    # D38 / AUDIT-SENS S2: the neutral control loses an equal or larger share.
    ('part of the mid-layer loss is persona-specific', 'S2: persona-specific loss withdrawn'),
    # D36 / H1-TEST: the RepE-LM test split was evaluated once on 2026-11-13.
    ('test protocol is frozen but has produced no results', 'H1-TEST was run on 2026-11-13'),
    ('to evaluate the frozen repe-lm test protocol', 'H1-TEST was run on 2026-11-13'),
)

# Words that, in the same sentence, mark a claim as historical rather than asserted.
HISTORICAL_MARKERS = ('withdrawn', 'withdraw', 'retracted', 'superseded', 'historical', 'invalid', 'no longer',
                      'earlier version', 'earlier draft', 'previously claimed')

_LATEX_COMMENT = re.compile(r'(?<!\\)%[^\n]*')
_LATEX_COMMAND = re.compile(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?')
_SPACING = re.compile(r'\\[,;:! ]|~')
_MARKUP = re.compile(r'[{}*`]')
_WHITESPACE = re.compile(r'\s+')
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')


def normalise(text: str) -> str:
    """Lower-case text with LaTeX comments, commands, spacing and Markdown emphasis removed; single spaces."""
    text = _LATEX_COMMENT.sub(' ', text)
    text = _SPACING.sub(' ', text)
    text = _LATEX_COMMAND.sub(' ', text)
    text = _MARKUP.sub(' ', text)
    return _WHITESPACE.sub(' ', text).strip().lower()


def asserted_hits(text: str, phrase: str) -> list:
    """Sentences that contain `phrase` without marking it as historical."""
    return [s for s in _SENTENCE_END.split(normalise(text))
            if phrase in s and not any(m in s for m in HISTORICAL_MARKERS)]


def _scanned_files():
    files = []
    for pattern in SCANNED_GLOBS:
        files.extend(sorted(ROOT.glob(pattern)))
    return files


def test_some_files_are_scanned():
    assert _scanned_files(), 'no paper or curated documents found; check SCANNED_GLOBS'


@pytest.mark.parametrize('phrase,reason', WITHDRAWN)
def test_withdrawn_claim_absent(phrase, reason):
    hits = {str(f.relative_to(ROOT)): asserted_hits(f.read_text(errors='ignore'), phrase) for f in _scanned_files()}
    hits = {k: v for k, v in hits.items() if v}
    assert not hits, f'withdrawn claim "{phrase}" ({reason}) asserted in: {list(hits)}'


# --- self-tests of the matcher -------------------------------------------------------------------------------

def test_matcher_survives_latex_and_rewrapping():
    tex = ('The frozen collapse \\emph{is largely}\n   perturbation~sensitivity.\n'
           'Most of the apparent collapse therefore reflects the sensitivity of\n%  a comment\nthat reader to any '
           '\\textbf{perturbation} of this size.')
    assert asserted_hits(tex, 'is largely perturbation sensitivity')
    assert asserted_hits(tex, 'reflects the sensitivity of that reader to any perturbation')


def test_matcher_survives_markdown_emphasis():
    md = 'Same-sized random noise caused almost the same drop, *so the probe was reacting\nto any disturbance*.'
    assert asserted_hits(md, 'so the probe was reacting to any disturbance')


def test_matcher_allows_historical_mentions():
    tex = ('An earlier draft said the frozen collapse is largely perturbation sensitivity; this reading was '
           'withdrawn (D38). The frozen GoT collapse is largely perturbation sensitivity.')
    hits = asserted_hits(tex, 'is largely perturbation sensitivity')
    assert len(hits) == 1 and 'withdrawn' not in hits[0]


def test_matcher_ignores_latex_comments():
    assert not asserted_hits('% the frozen truth loss is mostly reader fragility\nText.',
                             'the frozen truth loss is mostly reader fragility')
