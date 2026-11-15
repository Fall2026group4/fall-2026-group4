#!/usr/bin/env python3
"""Build the editable GPT-2 SAE-faithfulness pipeline diagram (2026-11-13; Review #5-#8 F8).

Writes two files from ONE node/edge list, so they cannot drift apart:
  * gpt2_sae_faithfulness_pipeline.drawio  (uncompressed mxGraph XML; open and edit in diagrams.net / draw.io)
  * gpt2_sae_faithfulness_pipeline.svg     (vector export for papers and slides)
Every box corresponds to code in the repository (source given in the node's tooltip and in the README table):
  hook site and BOS exclusion  scripts/shared/spliced_reconstruction_evaluation.py (hook_name, Splicer.logprobs)
  SAE forward                  src/sae/modern_sae.py (pre, activate, decode); scripts/openwebtext/03_training_sae.py
  reconstruction metrics       src/evaluation/faithfulness_metrics.py (reconstruction_metrics)
  frozen / re-fit probes, PCA  scripts/shared/m2_four_level_faithfulness.py, src/evaluation/information_levels.py
  splicing ΔCE / KL            scripts/shared/spliced_reconstruction_evaluation.py (owt_eval)
  causal decomposition         scripts/repe/24_sae_patch_decomposition.py (decompose)

Run from the repository root: python architecture/diagrams/build_gpt2_pipeline_diagram.py
"""
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent
W, H = 236, 64
# id, label (lines split by \n), x, y, fill, source
NODES = [
    ('data', 'Data\nOWT · GoT (cities+neg_cities)\nRepE-LM persona pairs', 20, 40, '#eef3fb', 'data/'),
    ('gpt2', 'GPT-2 Small (frozen)\nlayers 0 … L−1', 290, 40, '#eef3fb', 'transformer_lens HookedTransformer'),
    ('hook', 'x = blocks.L.hook_resid_pre\n(BOS excluded)', 560, 40, '#fff6e5',
     'spliced_reconstruction_evaluation.py: hook_name, Splicer.logprobs'),
    ('enc', 'Encoder\npre = W_enc (x − b_dec) + b_enc', 560, 150, '#e9f6ee', 'src/sae/modern_sae.py: pre'),
    ('act', 'Activation → z\nReLU · TopK(k=128) · BatchTopK · JumpReLU', 560, 250, '#e9f6ee',
     'src/sae/modern_sae.py: activate'),
    ('dec', 'Decoder\nx̂ = W_dec z + b_dec ;  e = x − x̂', 560, 350, '#e9f6ee', 'src/sae/modern_sae.py: decode'),
    ('rest', 'GPT-2 layers L … 11\n→ next-token distribution', 290, 460, '#eef3fb', 'Splicer.logprobs'),
    ('m_rec', '1 Reconstruction\nNMSE · EV · cosine · L0', 860, 150, '#f3eefb',
     'faithfulness_metrics.reconstruction_metrics'),
    ('m_info', '2 Information\nfrozen probe on x̂ · re-fit probe\nnoise and PCA references', 860, 250, '#f3eefb',
     'm2_four_level_faithfulness.evaluate_source'),
    ('m_beh', '3 Behaviour (splice x̂)\nΔCE · KL(p_orig ‖ p_spliced)', 860, 350, '#f3eefb',
     'spliced_reconstruction_evaluation.owt_eval'),
    ('m_cau', '4 Causal (RepE-LM)\nΔ = Δ_rec + Δ_err ; T_rec / T_full', 860, 460, '#f3eefb',
     '24_sae_patch_decomposition.decompose'),
]
EDGES = [('data', 'gpt2', ''), ('gpt2', 'hook', ''), ('hook', 'enc', ''), ('enc', 'act', ''), ('act', 'dec', ''),
         ('dec', 'rest', 'x̂ replaces x'), ('dec', 'm_rec', ''), ('dec', 'm_info', ''), ('dec', 'm_beh', ''),
         ('dec', 'm_cau', ''), ('hook', 'm_info', 'probes fitted on original x')]
POS = {n[0]: n for n in NODES}


def drawio():
    cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
    for nid, label, x, y, fill, src in NODES:
        v = escape(label).replace('\n', '&lt;br&gt;')
        cells.append(f'<UserObject id="{nid}" label="{v}" tooltip="{escape(src)}">'
                     f'<mxCell style="rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor=#5b6577;fontSize=11;"'
                     f' vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{W}" height="{H}" as="geometry"/>'
                     f'</mxCell></UserObject>')
    for i, (a, b, lab) in enumerate(EDGES):
        cells.append(f'<mxCell id="e{i}" value="{escape(lab)}" style="endArrow=block;html=1;fontSize=10;'
                     f'strokeColor=#5b6577;" edge="1" parent="1" source="{a}" target="{b}">'
                     f'<mxGeometry relative="1" as="geometry"/></mxCell>')
    return ('<mxfile host="build_gpt2_pipeline_diagram.py"><diagram name="GPT-2 SAE faithfulness pipeline">'
            '<mxGraphModel dx="1000" dy="600" grid="1" gridSize="10" page="1"><root>' + ''.join(cells) +
            '</root></mxGraphModel></diagram></mxfile>\n')


def anchor(a, b):
    """Edge from the side of a facing b to the facing side of b."""
    _, _, ax, ay, *_ = POS[a]
    _, _, bx, by, *_ = POS[b]
    if abs(bx - ax) > abs(by - ay):
        return (ax + W, ay + H / 2, bx, by + H / 2) if bx > ax else (ax, ay + H / 2, bx + W, by + H / 2)
    return (ax + W / 2, ay + H, bx + W / 2, by) if by > ay else (ax + W / 2, ay, bx + W / 2, by + H)


def svg():
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="550" viewBox="0 0 1120 550" '
           'font-family="Helvetica, Arial, sans-serif">',
           '<title>GPT-2 SAE faithfulness pipeline</title>',
           '<defs><marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
           '<path d="M0,0 L8,4 L0,8 z" fill="#5b6577"/></marker></defs>',
           '<rect width="1120" height="550" fill="#ffffff"/>']
    for a, b, lab in EDGES:
        x1, y1, x2, y2 = anchor(a, b)
        out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#5b6577" stroke-width="1.4" '
                   f'marker-end="url(#arr)"/>')
        if lab:
            f_ = 0.3 if a == 'hook' else 0.5          # place the long probe label near its start
            out.append(f'<text x="{x1 + f_ * (x2 - x1) + 6}" y="{y1 + f_ * (y2 - y1)}" font-size="10" fill="#333">'
                       f'{escape(lab)}</text>')
    for nid, label, x, y, fill, src in NODES:
        out.append(f'<g><title>{escape(src)}</title><rect x="{x}" y="{y}" width="{W}" height="{H}" rx="8" '
                   f'fill="{fill}" stroke="#5b6577"/>')
        lines = label.split('\n')
        for j, ln in enumerate(lines):
            yy = y + H / 2 + (j - (len(lines) - 1) / 2) * 14 + 4
            out.append(f'<text x="{x + W / 2}" y="{yy}" font-size="11" text-anchor="middle" fill="#111"'
                       f'{" font-weight=" + chr(34) + "bold" + chr(34) if j == 0 else ""}>{escape(ln)}</text>')
        out.append('</g>')
    out.append('</svg>\n')
    return '\n'.join(out)


def main():
    for name, text in (('gpt2_sae_faithfulness_pipeline.drawio', drawio()),
                       ('gpt2_sae_faithfulness_pipeline.svg', svg())):
        p = OUT / name
        if p.exists():
            raise SystemExit(f'{p} exists; refusing to overwrite.')
        p.write_text(text)
        print('wrote', p.relative_to(OUT.parents[1]))


if __name__ == '__main__':
    main()
