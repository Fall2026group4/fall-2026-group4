#!/usr/bin/env python3
"""TSP-R Phase 0: fresh Geometry-of-Truth families. Integrity, leakage checks, locked DEV/TEST split,
cities cross-fitting folds, and the DEV-only truth-signal eligibility check. No SAE is trained.

Stages (run from the repository root, in order):
    python -I scripts/geometry_of_truth/29_tsp_r_phase0_fresh_families.py prepare
        Text-level only: verifies the pinned downloads, runs integrity and leakage checks, writes the
        DEV/TEST split and the cities folds, and writes preregistration.json. No activation and no TEST
        label is used. It prints label balance for DEV rows only.
    (commit the preregistration)
    python -I scripts/geometry_of_truth/29_tsp_r_phase0_fresh_families.py eligibility
        GPT-2 L8/L11 activations for DEV rows only. Grouped-CV logistic AUROC (frozen rule: eligible if
        L8 AUROC >= 0.70), permuted-label and surface baselines, R2-base reconstruction headroom (inference
        only), and a precision proxy for Phase 1. TEST rows are never tokenised, embedded or scored.

Each stage refuses to overwrite its outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
COMMIT = '5d1c630c44f7e50bda7ad86d601ccadf9abc5ddb'           # saprmarks/geometry-of-truth main, 2024-08-08
RAW = REPO / f'data/got/fresh/raw_{COMMIT}'
SPLITS = REPO / 'data/got/fresh/splits'
ACTS = REPO / 'data/got/activations/gpt2_small/fresh_phase0'
OUT = REPO / 'results/geometry_of_truth/truth_subspace_preservation/replication_phase0'
R2 = REPO / 'results/openwebtext/r2_scaled_saes/checkpoints/layer_8'
SALT = 'TSP-R-phase0-2026-11-14'
DEV_FRAC = 0.30
ELIGIBILITY_AUROC = 0.70
LAYERS = (8, 11)
NEAR_DUP_COS = 0.90
C_GRID = (0.01, 0.1, 1.0)
N_BOOT, BSEED = 2000, 777
R2_SEEDS = (42, 1042, 2042)

# Upstream git blob SHA-1 for each file at COMMIT (from the GitHub contents API, 2026-11-14).
UPSTREAM_BLOB = {
    'sp_en_trans.csv': 'f99aa08658917c9000b6db1598458fa29ae4aeea',
    'neg_sp_en_trans.csv': '60dd635566cec80004d95cf65a5138da00b4788e',
    'companies_true_false.csv': '403a4a911e9ca4421e6d9dfe7db77b931f4db8a6',
    'common_claim_true_false.csv': '455c67966cac0178035ec7d8e62d7b5cfd9d4779',
    'counterfact_true_false.csv': '7b87800bffb82421accf3033016a1e1826ca2dc9',
    'counterfact.json': '34be0e5617d9d4b884fddb197e0f7769d1a31ad3',
    'cities_cities_conj.csv': '83fe4da613b308269ffd74080c11c055eb2f5424',
    'cities_cities_disj.csv': '0dd6fdbd95b34c2182efb1c26f254a574611ae71',
    'make_conj_disj.py.txt': '8f2ed2a7775278fe09ce2d529ae155e3e62db140',
}
# family -> (file, role). 'independent' families are candidates for P2; 'secondary' share entities with cities.
FAMILIES = {
    'sp_en_trans': ('sp_en_trans.csv', 'independent'),
    'neg_sp_en_trans': ('neg_sp_en_trans.csv', 'independent'),
    'companies_true_false': ('companies_true_false.csv', 'independent'),
    'common_claim_true_false': ('common_claim_true_false.csv', 'independent'),
    'counterfact_true_false': ('counterfact_true_false.csv', 'independent'),
    'cities_cities_conj': ('cities_cities_conj.csv', 'secondary'),
    'cities_cities_disj': ('cities_cities_disj.csv', 'secondary'),
}
# Existing project text that fresh rows must not duplicate.
PROJECT_TEXT = {
    'got_cities': ('data/got/cities.csv', 'statement'),
    'got_neg_cities': ('data/got/neg_cities.csv', 'statement'),
    'got_larger_than': ('data/got/larger_than.csv', 'statement'),
    'got_smaller_than': ('data/got/smaller_than.csv', 'statement'),
    'repe_facts': ('data/repe/facts_clean.csv', 'statement'),
    'repe_cross_wording': ('data/repe/cross_wording_validation.csv', 'statement'),
}


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def blob(p):
    return subprocess.run(['git', 'hash-object', str(p)], capture_output=True, text=True, check=True).stdout.strip()


def norm(s):
    s = unicodedata.normalize('NFKC', str(s)).lower().strip()
    s = re.sub(r'\s+', ' ', s)
    return re.sub(r'[\s.!?]+$', '', s)


def unit(*parts):
    """Deterministic uniform number in [0, 1) from a salted SHA-256."""
    h = hashlib.sha256('|'.join((SALT,) + tuple(map(str, parts))).encode()).hexdigest()
    return int(h[:15], 16) / 16 ** 15


def refuse(*paths):
    for p in paths:
        if Path(p).exists():
            sys.exit(f'refusing to overwrite {p}')


# ------------------------------------------------------------------------------------------------ groups
SP = re.compile(r"^The Spanish word '(.+?)' (?:does not mean|means) '")
CO = re.compile(r'^(.+?) (?:operates in the industry of|has headquarters in|engages in) ')
CO2 = re.compile(r'^(.+?) (?:is|operates as|was) ')


def counterfact_components(df):
    """Union-find over templates and Wikidata relation ids: templates shared across relations are merged."""
    cf = json.loads((RAW / 'counterfact.json').read_text())
    rel_of = defaultdict(set)
    for r in cf:
        rel_of[r['requested_rewrite']['prompt']].add(r['requested_rewrite']['relation_id'])
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for t in df.relation.unique():
        assert rel_of[t], f'template without relation id: {t}'
        for r in rel_of[t]:
            parent[find('T:' + t)] = find('R:' + r)
    comp = {}
    for t in df.relation.unique():
        rels = sorted({r for tt in df.relation.unique() if find('T:' + tt) == find('T:' + t) for r in rel_of[tt]})
        comp[t] = '+'.join(rels)
    return df.relation.map(comp), df.relation.map(lambda t: ','.join(sorted(rel_of[t])))


def load_family(name):
    f, role = FAMILIES[name]
    df = pd.read_csv(RAW / f)
    df.insert(0, 'row_id', np.arange(len(df)))
    if name in ('sp_en_trans', 'neg_sp_en_trans'):
        df['group_id'] = 'sp_en:' + df.statement.map(lambda s: SP.match(s).group(1))
        df['entity'] = df.statement.map(lambda s: SP.match(s).group(1))
    elif name == 'companies_true_false':
        names = {CO.match(s).group(1) for s in df.statement if CO.match(s)}

        def company(s):
            if CO.match(s):
                return CO.match(s).group(1)
            hit = [n for n in names if s.startswith(n + ' ')]
            if hit:
                return max(hit, key=len)
            return CO2.match(s).group(1)

        df['entity'] = df.statement.map(company)
        df['group_id'] = 'company:' + df.entity
    elif name == 'common_claim_true_false':
        df['entity'] = ''
        df['group_id'] = 'claim:' + df.statement.map(lambda s: hashlib.sha256(norm(s).encode()).hexdigest()[:16])
    elif name == 'counterfact_true_false':
        df['component'], df['relation_ids'] = counterfact_components(df)
        df['entity'] = df.subject.astype(str)
        df['group_id'] = 'relation:' + df.component
    else:                                                           # conj / disj
        df['entity'] = df.city1.astype(str) + '|' + df.city2.astype(str)
        df['group_id'] = 'pair:' + df.apply(lambda r: '|'.join(sorted([str(r.city1), str(r.city2)])), axis=1)
    df['role'] = role
    return df


# ------------------------------------------------------------------------------------------------ prepare
def integrity(name, df):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('gpt2')
    lens = np.array([len(tok(s)['input_ids']) + 1 for s in df.statement])      # + BOS
    n = norm
    return {
        'family': name, 'role': FAMILIES[name][1], 'rows': len(df),
        'null_statement': int(df.statement.isna().sum()), 'null_label': int(df.label.isna().sum()),
        'label_values_binary': bool(df.label.isin([0, 1]).all()),               # value check only, no balance
        'empty_statement': int((df.statement.astype(str).str.strip() == '').sum()),
        'exact_duplicate_statements': int(df.statement.duplicated().sum()),
        'normalised_duplicate_statements': int(df.statement.map(n).duplicated().sum()),
        'non_ascii_rows': int(df.statement.map(lambda s: not str(s).isascii()).sum()),
        'ends_with_period': float(df.statement.str.rstrip().str.endswith('.').mean()),
        'groups': int(df.group_id.nunique()), 'entities': int(df.entity.nunique()),
        'tokens_mean': float(lens.mean()), 'tokens_max': int(lens.max()),
    }


def project_corpus():
    rows = []
    for src, (p, col) in PROJECT_TEXT.items():
        d = pd.read_csv(REPO / p)
        rows += [(src, s) for s in d[col].dropna().astype(str).unique()]
    return pd.DataFrame(rows, columns=['source', 'statement'])


def leakage(df, corpus, city_split):
    """Flags per row. Text only: labels are not read."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    proj_norm = dict(zip(corpus.statement.map(norm), corpus.source))
    df['exact_overlap_project'] = df.statement.map(lambda s: proj_norm.get(norm(s), ''))
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(4, 4), sublinear_tf=True)
    P = vec.fit_transform(corpus.statement.map(norm))
    best, src = np.zeros(len(df)), np.empty(len(df), dtype=object)
    for b in range(0, len(df), 2000):
        F = vec.transform(df.statement.iloc[b:b + 2000].map(norm))
        S = (F @ P.T).toarray()
        best[b:b + 2000] = S.max(1)
        src[b:b + 2000] = corpus.source.to_numpy()[S.argmax(1)]
    df['max_cos_project'] = best.round(4)
    df['near_overlap_project'] = np.where(best >= NEAR_DUP_COS, src, '')
    # Whole-word, case-sensitive mentions of any cities.csv city (the TSP training entity set).
    names = sorted(city_split, key=len, reverse=True)
    pat = re.compile(r'(?<![\w])(' + '|'.join(map(re.escape, names)) + r')(?![\w])')
    hits = df.statement.map(lambda s: sorted(set(pat.findall(str(s)))))
    df['project_cities_mentioned'] = hits.map(lambda h: ';'.join(h))
    df['mentions_exposed_test_city'] = hits.map(lambda h: any(city_split[c] == 'test' for c in h))
    df['mentions_project_city'] = hits.map(bool)
    return df


def assign_split(name, df):
    if name == 'counterfact_true_false':
        # Hold out whole relation components; DEV = components in salted-hash order until >= 30 % of rows.
        sizes = df.groupby('group_id').size()
        order = sorted(sizes.index, key=lambda g: unit('counterfact', g))
        dev, n = set(), 0
        for g in order:
            if n >= DEV_FRAC * len(df):
                break
            dev.add(g)
            n += sizes[g]
        df['split'] = np.where(df.group_id.isin(dev), 'DEV', 'TEST')
        # Entity guard: a TEST row whose subject also occurs in DEV is excluded.
        dev_subj = set(df.subject[df.split == 'DEV'])
        df['subject_in_dev'] = (df.split == 'TEST') & df.subject.isin(dev_subj)
    else:
        key = 'sp_en' if name in ('sp_en_trans', 'neg_sp_en_trans') else name
        df['split'] = np.where(df.group_id.map(lambda g: unit(key, g) < DEV_FRAC), 'DEV', 'TEST')
        df['subject_in_dev'] = False
    if name in ('cities_cities_conj', 'cities_cities_disj'):
        # Entity-disjoint guard on single cities: a TEST pair sharing a city with any DEV pair is excluded.
        dev_c = set(df.city1[df.split == 'DEV']) | set(df.city2[df.split == 'DEV'])
        df['subject_in_dev'] = (df.split == 'TEST') & (df.city1.isin(dev_c) | df.city2.isin(dev_c))
    # Exclusion reasons (text-level, decided before any activation): applied to DEV and TEST alike.
    reason = np.full(len(df), '', dtype=object)
    # v2 (before freeze, before any activation): every repeated normalised statement is dropped, text only.
    # v1 found label-conflicting duplicates in companies (DEV); a text-only rule avoids reading TEST labels.
    reason[df.statement.map(norm).duplicated(keep=False).to_numpy()] = 'duplicate_statement_within_family'
    reason[df.subject_in_dev.to_numpy()] = 'entity_in_dev'
    reason[df.mentions_exposed_test_city.to_numpy()] = 'exposed_test_city'
    reason[(df.near_overlap_project != '').to_numpy()] = 'near_duplicate_of_project_text'
    reason[(df.exact_overlap_project != '').to_numpy()] = 'exact_duplicate_of_project_text'
    df['excluded'] = reason
    # P2 primary also drops rows mentioning any TSP-training city (independent families only).
    df['p2_primary'] = (df.role == 'independent') & (df.excluded == '') & ~df.mentions_project_city
    return df


def cities_folds():
    sp = pd.read_csv(REPO / 'data/got/splits/cities_splits.csv')
    ab = pd.read_csv(REPO / 'data/got/splits/train_AB_groups_truth_aware_pilot.csv')
    ab = dict(zip(ab[ab.domain == 'geographic'].group_id, ab[ab.domain == 'geographic'].half))
    g = sp.groupby('group_id').split.first()
    test = set(g[g == 'test'].index)
    usable = sorted(g[g != 'test'].index, key=lambda c: unit('cities_folds', c))
    folds = pd.DataFrame({'group_id': usable, 'fold': [i % 3 for i in range(len(usable))]})
    folds['previous_split'] = folds.group_id.map(g)
    folds['previous_pilot_half'] = np.where(folds.previous_split == 'val', 'V', folds.group_id.map(ab))
    roles = {f: {'V': f, 'B': (f + 1) % 3, 'A': (f + 2) % 3} for f in range(3)}
    return folds, test, roles


def prepare():
    out_files = [SPLITS, OUT / 'preregistration.json', OUT / 'integrity.csv']
    refuse(*out_files)
    t0 = time.time()
    # 1. Provenance.
    manifest = []
    for f, b in UPSTREAM_BLOB.items():
        p = RAW / f
        got = blob(p)
        assert got == b, f'{f}: blob {got} != upstream {b}'
        manifest.append({'file': f, 'upstream_path': f'datasets/{f.replace(".py.txt", ".py")}', 'git_blob_sha1': got,
                         'sha256': sha256(p), 'bytes': p.stat().st_size})
    existing = []
    for f in ('cities.csv', 'neg_cities.csv', 'larger_than.csv', 'smaller_than.csv'):
        existing.append({'file': f'data/got/{f}', 'git_blob_sha1': blob(REPO / 'data/got' / f)})
    # 2. Cities folds and city -> split map.
    folds, test_cities, roles = cities_folds()
    cities = pd.read_csv(REPO / 'data/got/cities.csv')
    cs = pd.read_csv(REPO / 'data/got/splits/cities_splits.csv').groupby('group_id').split.first()
    city_split = {c: cs.get(c, 'unknown') for c in cities.city.unique()}
    corpus = project_corpus()
    # 3. Families.
    SPLITS.mkdir(parents=True)
    integ, counts, dev_balance, leak = [], [], [], []
    fold_of = dict(zip(folds.group_id, folds.fold))
    conj_fold = []
    for name in FAMILIES:
        df = load_family(name)
        integ.append(integrity(name, df))
        df = leakage(df, corpus, city_split)
        df = assign_split(name, df)
        keep = ['row_id', 'group_id', 'entity', 'split', 'excluded', 'p2_primary', 'mentions_project_city',
                'project_cities_mentioned', 'max_cos_project']
        if name == 'counterfact_true_false':
            keep.insert(3, 'relation_ids')
        sp = df[keep].copy()
        sp['statement_sha256_16'] = df.statement.map(lambda s: hashlib.sha256(str(s).encode()).hexdigest()[:16])
        sp.to_csv(SPLITS / f'{name}_splits.csv', index=False)                   # no labels in split files
        for s in ('DEV', 'TEST'):
            m = df.split == s
            counts.append({'family': name, 'role': FAMILIES[name][1], 'split': s, 'rows': int(m.sum()),
                           'groups': int(df.group_id[m].nunique()), 'entities': int(df.entity[m].nunique()),
                           'usable_rows': int((m & (df.excluded == '')).sum()),
                           'usable_entities': int(df.entity[m & (df.excluded == '')].nunique()),
                           'p2_primary_rows': int((m & df.p2_primary).sum()),
                           'p2_primary_entities': int(df.entity[m & df.p2_primary].nunique()),
                           **{f'excl_{r}': int((m & (df.excluded == r)).sum()) for r in
                              ('duplicate_statement_within_family', 'entity_in_dev', 'exposed_test_city', 'near_duplicate_of_project_text',
                               'exact_duplicate_of_project_text')},
                           'mentions_project_city': int((m & df.mentions_project_city).sum())})
        d = df[(df.split == 'DEV') & (df.excluded == '')]                      # DEV labels only
        dev_balance.append({'family': name, 'dev_usable_rows': len(d), 'dev_frac_true': round(d.label.mean(), 4),
                            'dev_label_conflicting_duplicates': int(
                                (d.groupby(d.statement.map(norm)).label.nunique() > 1).sum())})
        leak.append({'family': name, 'rows': len(df),
                     'exact_overlap_project': int((df.exact_overlap_project != '').sum()),
                     'near_overlap_project_cos>=0.9': int((df.near_overlap_project != '').sum()),
                     'max_cos_project_p99': round(float(np.quantile(df.max_cos_project, 0.99)), 4),
                     'max_cos_project_max': round(float(df.max_cos_project.max()), 4),
                     'near_overlap_sources': ';'.join(sorted(set(df.near_overlap_project) - {''})),
                     'mentions_project_city': int(df.mentions_project_city.sum()),
                     'mentions_exposed_test_city': int(df.mentions_exposed_test_city.sum()),
                     'dev_test_group_overlap': len(set(df.group_id[df.split == 'DEV']) & set(df.group_id[df.split == 'TEST'])),
                     'dev_test_entity_overlap_after_exclusion': len(
                         set(df.entity[(df.split == 'DEV') & (df.excluded == '')]) &
                         set(df.entity[(df.split == 'TEST') & (df.excluded == '')]))})
        if FAMILIES[name][1] == 'secondary':
            u = df[df.excluded == '']
            f1, f2 = u.city1.map(fold_of), u.city2.map(fold_of)
            for s in ('DEV', 'TEST'):
                for f in range(3):
                    conj_fold.append({'family': name, 'split': s, 'fold_as_V': f,
                                      'rows_both_cities_in_V': int(((u.split == s) & (f1 == f) & (f2 == f)).sum())})
        print(f'{name}: {len(df)} rows, {time.time() - t0:.0f}s', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    folds.to_csv(SPLITS / 'cities_crossfit_folds.csv', index=False)
    pd.DataFrame(integ).to_csv(OUT / 'integrity.csv', index=False)
    pd.DataFrame(counts).to_csv(OUT / 'split_counts.csv', index=False)
    pd.DataFrame(dev_balance).to_csv(OUT / 'dev_label_balance.csv', index=False)
    pd.DataFrame(leak).to_csv(OUT / 'leakage.csv', index=False)
    pd.DataFrame(conj_fold).to_csv(OUT / 'conj_disj_fold_counts.csv', index=False)
    fold_summary = folds.groupby('fold').agg(groups=('group_id', 'size'),
                                             prev_val=('previous_split', lambda s: int((s == 'val').sum())))
    split_hashes = {p.name: sha256(p) for p in sorted(SPLITS.glob('*.csv'))}
    prereg = {
        'experiment': 'TSP-R Phase 0 (DEV-only eligibility of fresh GoT families)',
        'date': '2026-11-14', 'status': 'FROZEN before any fresh-family activation was computed',
        'source': {'repo': 'github.com/saprmarks/geometry-of-truth', 'commit': COMMIT, 'files': manifest,
                   'existing_project_files_match_upstream': existing,
                   'license': 'no LICENSE file in the repository (GitHub API license: null); raw files are '
                              'kept locally, gitignored, not redistributed; academic research use only'},
        'split': {'salt': SALT, 'dev_fraction': DEV_FRAC,
                  'unit': {'sp_en_trans/neg_sp_en_trans': 'Spanish word (shared across both files)',
                           'companies_true_false': 'company name',
                           'common_claim_true_false': 'normalised statement',
                           'counterfact_true_false': 'Wikidata relation component (templates shared by relations '
                                                     'merged); DEV = components in hash order until >= 30 % rows; '
                                                     'TEST rows whose subject is in DEV excluded',
                           'cities_cities_conj/disj': 'unordered city pair; TEST pairs sharing a city with DEV '
                                                      'excluded'},
                  'amendment': 'v2: rows whose normalised statement repeats within the family are excluded '
                               '(text only). v1 outputs kept in prepare_v1_superseded/; no activation had been '
                               'computed',
                  'exclusions_all_splits': ['repeated normalised statement within the family', 'exact or near (char-4gram tf-idf cos >= 0.90) duplicate of project '
                                            'text', 'mentions one of the 113 exposed cities TEST cities'],
                  'p2_primary_additionally_excludes': 'rows mentioning any cities.csv city (TSP training '
                                                      'entities); included in a sensitivity analysis',
                  'split_file_sha256': split_hashes},
        'cities_crossfit': {'groups': int(len(folds)), 'excluded_exposed_test_groups': len(test_cities),
                            'fold_sizes': fold_summary.groups.to_dict(),
                            'roles': {f'fold{f}': r for f, r in roles.items()},
                            'note': 'fold f: V = f (evaluation), B = f+1 (probes), A = f+2 (S and GoT replay)'},
        'eligibility_rule': {
            'metric': 'original-activation logistic AUROC, GPT-2 L8 blocks.8.hook_resid_pre final token, '
                      'pooled out-of-fold scores of GroupKFold(5) over the split unit, C chosen by inner '
                      f'GroupKFold(3) over {list(C_GRID)}; DEV usable rows only',
            'threshold': f'eligible for P2 iff L8 AUROC >= {ELIGIBILITY_AUROC} on the P2-primary DEV rows',
            'reported_not_decisive': ['L11 AUROC', 'entity-level GroupKFold AUROC (counterfact)',
                                      'permuted-label AUROC', 'surface tf-idf word 1-2gram logistic AUROC',
                                      'bootstrap 95 % CI over entities',
                                      'R2 base (seeds 42/1042/2042) re-fit and frozen gaps, NMSE, L0 on DEV',
                                      'precision proxy and minimal detectable effect for P2'],
            'surface_flag': 'a family is flagged SURFACE-SOLVABLE if the surface AUROC >= 0.70 (reported; it '
                            'does not change eligibility)'},
        'not_used': ['TEST labels', 'TEST activations', 'any TSP pilot fine-tune (no tsp/ctrl result on fresh '
                     'data in Phase 0)', 'cities TEST rows'],
        'script_sha256': sha256(Path(__file__)),
    }
    (OUT / 'preregistration.json').write_text(json.dumps(prereg, indent=2, default=int))
    print(json.dumps({'counts': counts}, default=int)[:200])
    print(f'prepare done in {time.time() - t0:.0f}s')


# ------------------------------------------------------------------------------------------------ eligibility
def extract_dev(model, name, df, device):
    import torch
    out = ACTS / f'{name}_DEV_final_token.pt'
    if out.exists():
        d = torch.load(out, map_location='cpu', weights_only=False)
        assert np.array_equal(d['row_ids'].numpy(), df.row_id.to_numpy())
        return {L: d['activations'][L].numpy() for L in LAYERS}
    hooks = [f'blocks.{L}.hook_resid_pre' for L in LAYERS]
    got = {L: [] for L in LAYERS}
    for b in range(0, len(df), 32):
        seqs = [model.to_tokens(str(s), prepend_bos=True, move_to_device=False)[0] for s in df.statement.iloc[b:b + 32]]
        lens = torch.tensor([len(s) for s in seqs])
        pad = torch.full((len(seqs), int(lens.max())), model.tokenizer.eos_token_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            pad[i, :len(s)] = s
        with torch.inference_mode():
            _, cache = model.run_with_cache(pad.to(device), names_filter=lambda n: n in hooks, return_type=None)
        idx = torch.arange(len(seqs))
        for L, h in zip(LAYERS, hooks):
            got[L].append(cache[h][idx, (lens - 1).to(device)].float().cpu())
    acts = {L: torch.cat(v) for L, v in got.items()}
    ACTS.mkdir(parents=True, exist_ok=True)
    torch.save({'family': name, 'split': 'DEV', 'model': 'gpt2-small', 'position': 'final_nonpadding_token',
                'prepend_bos': True, 'row_ids': torch.as_tensor(df.row_id.to_numpy()), 'activations': acts}, out)
    return {L: v.numpy() for L, v in acts.items()}


def grouped_oof(X, y, g, folds=5, Xeval=None, scorer=None):
    """Pooled out-of-fold decision scores. If Xeval is given, the probe fitted on X is applied to Xeval (frozen)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def lr(C):
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, random_state=0))

    k = min(folds, len(set(g)))
    s, se = np.zeros(len(y)), (None if Xeval is None else [np.zeros(len(y)) for _ in Xeval])
    for a, b in GroupKFold(k).split(X, y, g):
        best = (-1, C_GRID[0])
        ki = min(3, len(set(g[a])))
        if ki >= 2:
            for C in C_GRID:
                sc = np.zeros(len(a))
                for aa, bb in GroupKFold(ki).split(X[a], y[a], g[a]):
                    sc[bb] = lr(C).fit(X[a][aa], y[a][aa]).decision_function(X[a][bb])
                v = roc_auc_score(y[a], sc) if len(set(y[a])) > 1 else 0.5
                if v > best[0] + 1e-12:
                    best = (v, C)
        m = lr(best[1]).fit(X[a], y[a])
        s[b] = m.decision_function(X[b])
        if Xeval is not None:
            for j, Xe in enumerate(Xeval):
                se[j][b] = m.decision_function(Xe[b])
    return s, se


def boot_auc(y, scores, clusters, idx):
    from sklearn.metrics import roc_auc_score
    out = []
    for ii in idx:
        out.append([roc_auc_score(y[ii], s[ii]) for s in scores])
    return np.array(out)


def eligibility():
    import torch
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from transformer_lens import HookedTransformer

    prereg = json.loads((OUT / 'preregistration.json').read_text())
    for f, h in prereg['split']['split_file_sha256'].items():
        assert sha256(SPLITS / f) == h, f'split file changed: {f}'
    refuse(OUT / 'eligibility.csv', OUT / 'headroom.csv', OUT / 'feasibility.json')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HookedTransformer.from_pretrained('gpt2-small', device=device)
    model.eval()
    sys.path.insert(0, str(REPO / 'scripts/geometry_of_truth'))
    saes = {}
    for s in R2_SEEDS:
        ck = torch.load(R2 / f'topk_seed{s}/checkpoint_final.pt', map_location='cpu', weights_only=False)
        st, cfg = ck['model_state_dict'], ck['config']
        assert cfg['architecture'] == 'topk' and cfg['format'] == 'modern_v1'
        saes[s] = (int(cfg['k']), *(st[k].to(device) for k in ('encoder.weight', 'encoder.bias', 'decoder.weight',
                                                                  'output_bias')), sha256(R2 / f'topk_seed{s}/checkpoint_final.pt'))

    def recon(s, X):
        k, We, be, Wd, b, _ = saes[s]
        H, L0 = [], []
        with torch.no_grad():
            for i in range(0, len(X), 4096):
                x = torch.as_tensor(X[i:i + 4096], device=device)
                pre = torch.relu((x - b) @ We.T + be)
                v, j = pre.topk(k, -1)
                z = torch.zeros_like(pre).scatter(-1, j, v)
                H.append((z @ Wd.T + b).cpu().numpy())
                L0.append((z > 0).sum(-1).float().cpu().numpy())
        return np.concatenate(H), np.concatenate(L0)

    rng = np.random.default_rng(BSEED)
    elig, head, timing = [], [], {}
    for name, (f, role) in FAMILIES.items():
        raw = pd.read_csv(RAW / f)
        sp = pd.read_csv(SPLITS / f'{name}_splits.csv')
        dev = sp[(sp.split == 'DEV') & (sp.excluded == '')].copy()
        assert (sp.loc[dev.index, 'split'] == 'DEV').all()
        dev['statement'] = raw.statement.to_numpy()[dev.row_id]
        dev['label'] = raw.label.to_numpy()[dev.row_id]                          # DEV labels only
        t0 = time.time()
        A = extract_dev(model, name, dev, device)
        timing[f'{name}_extract_s'] = round(time.time() - t0, 1)
        subsets = {'all_usable': np.ones(len(dev), bool)}
        if role == 'independent':
            subsets['p2_primary'] = dev.p2_primary.to_numpy().astype(bool)
        for sub, m in subsets.items():
            d, y = dev[m], dev.label.to_numpy()[m]
            g, ent = d.group_id.to_numpy(), d.entity.astype(str).to_numpy()
            if name == 'common_claim_true_false':
                ent = g
            ent_codes = pd.factorize(ent)[0]
            uniq = np.unique(ent_codes)
            members = [np.flatnonzero(ent_codes == u) for u in uniq]
            idx = [np.concatenate([members[j] for j in rng.integers(0, len(uniq), len(uniq))]) for _ in range(N_BOOT)]
            row = {'family': name, 'role': role, 'subset': sub, 'rows': int(m.sum()), 'split_units': len(set(g)),
                   'entities': len(uniq), 'frac_true': round(float(y.mean()), 4)}
            scores = {}
            for L in LAYERS:
                t0 = time.time()
                s, _ = grouped_oof(A[L][m], y, g)
                timing[f'{name}_{sub}_L{L}_probe_s'] = round(time.time() - t0, 1)
                scores[f'L{L}'] = s
                if name == 'counterfact_true_false':
                    scores[f'L{L}_entity_cv'], _ = grouped_oof(A[L][m], y, ent)
            perm = rng.permutation(y)
            sp_, _ = grouped_oof(A[8][m], perm, g)
            # Surface baseline: word 1-2gram tf-idf, same outer folds, fixed C = 1.
            surf = np.zeros(len(y))
            for a, b in GroupKFold(min(5, len(set(g)))).split(d, y, g):
                vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
                Xa = vec.fit_transform(d.statement.iloc[a])
                surf[b] = LogisticRegression(C=1.0, max_iter=5000).fit(Xa, y[a]).decision_function(
                    vec.transform(d.statement.iloc[b]))
            keys = list(scores)
            bt = boot_auc(y, [scores[k] for k in keys] + [surf], ent_codes, idx)
            for j, k in enumerate(keys):
                row[f'auroc_{k}'] = round(roc_auc_score(y, scores[k]), 4)
                row[f'auroc_{k}_ci'] = f'[{np.quantile(bt[:, j], .025):.3f}, {np.quantile(bt[:, j], .975):.3f}]'
            row['auroc_permuted_L8'] = round(roc_auc_score(perm, sp_), 4)
            row['auroc_surface'] = round(roc_auc_score(y, surf), 4)
            row['auroc_surface_ci'] = f'[{np.quantile(bt[:, -1], .025):.3f}, {np.quantile(bt[:, -1], .975):.3f}]'
            row['surface_solvable_flag'] = bool(row['auroc_surface'] >= 0.70)
            decisive = (sub == 'p2_primary')
            row['eligible_P2'] = bool(decisive and row['auroc_L8'] >= ELIGIBILITY_AUROC) if role == 'independent' else ''
            elig.append(row)
            # Headroom: R2 base reconstructions of DEV activations (inference only).
            X = A[8][m]
            recs = {s: recon(s, X) for s in R2_SEEDS}
            refit, frozen = {}, {}
            _, fz = grouped_oof(X, y, g, Xeval=[recs[s][0] for s in R2_SEEDS])
            for j, s in enumerate(R2_SEEDS):
                refit[s], _ = grouped_oof(recs[s][0], y, g)
                frozen[s] = fz[j]
            allsc = [scores['L8']] + [refit[s] for s in R2_SEEDS] + [frozen[s] for s in R2_SEEDS]
            bt = boot_auc(y, allsc, ent_codes, idx)
            base = bt[:, 0]
            for j, s in enumerate(R2_SEEDS):
                xh = recs[s][0]
                nmse = float(((xh - X) ** 2).sum() / ((X - X.mean(0)) ** 2).sum())
                gr, gf = base - bt[:, 1 + j], base - bt[:, 4 + j]
                head.append({'family': name, 'subset': sub, 'r2_seed': s, 'sae_sha256_16': saes[s][-1][:16],
                             'rows': int(m.sum()), 'nmse': round(nmse, 4), 'l0_mean': float(recs[s][1].mean()),
                             'auroc_orig': round(roc_auc_score(y, scores['L8']), 4),
                             'auroc_refit_xhat': round(roc_auc_score(y, refit[s]), 4),
                             'auroc_frozen_xhat': round(roc_auc_score(y, frozen[s]), 4),
                             'refit_gap': round(roc_auc_score(y, scores['L8']) - roc_auc_score(y, refit[s]), 4),
                             'refit_gap_boot_sd': round(float(gr.std()), 4),
                             'frozen_gap': round(roc_auc_score(y, scores['L8']) - roc_auc_score(y, frozen[s]), 4),
                             'frozen_gap_boot_sd': round(float(gf.std()), 4)})
            # Precision proxy: bootstrap sd of the paired difference in re-fit gap between two R2 seeds.
            d01 = (bt[:, 1] - bt[:, 2])
            d02 = (bt[:, 1] - bt[:, 3])
            elig[-1]['paired_refit_gap_diff_boot_sd_DEV'] = round(float(np.mean([d01.std(), d02.std()])), 4)
            print(f'{name}/{sub}: L8 {row["auroc_L8"]} L11 {row["auroc_L11"]} surface {row["auroc_surface"]} '
                  f'perm {row["auroc_permuted_L8"]}', flush=True)
    pd.DataFrame(elig).to_csv(OUT / 'eligibility.csv', index=False)
    pd.DataFrame(head).to_csv(OUT / 'headroom.csv', index=False)
    (OUT / 'feasibility.json').write_text(json.dumps({'timing_s': timing, 'device': device,
                                                       'script_sha256': sha256(Path(__file__))}, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['prepare', 'eligibility'])
    {'prepare': prepare, 'eligibility': eligibility}[ap.parse_args().stage]()
