"""RepE Stage 4 add-on: size-matched control for the honesty difference.

Honesty pair : same statement, honest vs lying persona        (matched_honest_lying)
Wording pairs: same statement, same honesty, different synonym (the control)
    honest wording : 'honest' vs 'truthful'  (honest prompts:    matched_honest_lying vs matched_truthful_lying)
    dishonest wording: 'lying' vs 'dishonest' (dishonest prompts: matched_honest_lying vs matched_honest_dishonest)
For each: median size of the difference, and median share of it lost in the SAE error.

Run from the repo root:  python scripts/run_repe_s4_control.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_repe_s4 as s4  # reuse the SAE loading and data code

A, B, C = "matched_honest_lying", "matched_truthful_lying", "matched_honest_dishonest"


def align(meta_a, meta_b, label):
    """Row indices in a and b for the same statements, both with the given label."""
    a = meta_a[meta_a["label"] == label].reset_index().set_index("statement")["index"]
    b = meta_b[meta_b["label"] == label].reset_index().set_index("statement")["index"]
    common = a.index.intersection(b.index)
    return a.loc[common].to_numpy(), b.loc[common].to_numpy()


def stats(d, dh):
    lost = np.linalg.norm(d - dh, axis=1) / (np.linalg.norm(d, axis=1) + 1e-8)
    return float(np.median(np.linalg.norm(d, axis=1))), float(np.median(lost))


def main():
    data = {n: s4.load_set(n) for n in (A, B, C)}
    ha, hb = align(data[A]["meta"], data[B]["meta"], 1)   # honest vs truthful
    da, dc = align(data[A]["meta"], data[C]["meta"], 0)   # lying vs dishonest
    h, u = data[A]["h"], data[A]["u"]
    print(f"honesty pairs {len(h)}, honest-wording pairs {len(ha)}, dishonest-wording pairs {len(da)}")

    rows = []
    for t, release in s4.RELEASES.items():
        for pos, sid in s4.sae_ids(release):
            sae = s4.load_sae(release, sid)
            x = {n: data[n]["x"][:, pos, :] for n in data}
            xh = {n: s4.run_sae(sae, x[n])[0] for n in data}
            hn, hl = stats(x[A][h] - x[A][u], xh[A][h] - xh[A][u])
            wn, wl = stats(x[A][ha] - x[B][hb], xh[A][ha] - xh[B][hb])
            dn, dl = stats(x[A][da] - x[C][dc], xh[A][da] - xh[C][dc])
            rows.append(dict(sae_type=t, position=pos,
                             honesty_size=hn, honesty_lost=hl,
                             honest_wording_size=wn, honest_wording_lost=wl,
                             dishonest_wording_size=dn, dishonest_wording_lost=dl))
            print(f"  {t} P{pos:02d}: honesty lost {hl:.2f} (size {hn:.1f})  |  wording lost "
                  f"{wl:.2f} (size {wn:.1f}) and {dl:.2f} (size {dn:.1f})")
            del sae

    df = pd.DataFrame(rows)
    s4.TAB_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(s4.TAB_DIR / "repe_s4_size_control.csv", index=False)
    print("\nSaved results/tables/repe/repe_s4_size_control.csv\n")
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()