#!/usr/bin/env python3
"""SAE faithfulness architecture diagram: one element list -> .drawio (editable) + .svg + .png.

Notation verified against the implementation:
  encoder / activation / decoder  scripts/openwebtext/03_training_sae.py (SAE.forward), src/sae/modern_sae.py
  loss and penalties              scripts/openwebtext/03_training_sae.py (train_one)
  EV, cosine, L0                  src/evaluation/faithfulness_metrics.py (reconstruction_metrics)
  splicing, KL(orig||spliced)     scripts/shared/spliced_reconstruction_evaluation.py
  causal restoration T(v)         scripts/repe/24_sae_patch_decomposition.py (decompose)
Run from the repository root: python architecture/diagrams/build_sae_architecture.py  (writes next to this script)
"""
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Rectangle

W, H = 1600, 1010
FG, BG = '#FFFFFF', '#000000'
FONT = 'Times New Roman'
NAME = 'sae_faithfulness_architecture'
PX = 1.3          # matplotlib pt at 100 dpi -> draw.io px

E = []          # element list shared by both renderers


def rect(x, y, w, h, lines=(), lw=1.4, fill=BG, dy=None):
    E.append(dict(k='rect', x=x, y=y, w=w, h=h, lines=list(lines), lw=lw, fill=fill, dy=dy))


def text(x, y, s, size=12, w=360, ha='center', bold=False):
    E.append(dict(k='text', x=x, y=y, s=s, size=size, w=w, ha=ha, bold=bold))


def poly(pts, label, direction, lw=1.6):
    E.append(dict(k='poly', pts=pts, label=label, direction=direction, lw=lw))


def arrow(pts, lw=1.4, head=True):
    E.append(dict(k='arrow', pts=pts, lw=lw, head=head))


def line(pts, lw=1.0):
    arrow(pts, lw=lw, head=False)


# ----------------------------------------------------------------- layout
text(800, 36, 'Does accurate SAE reconstruction preserve the information and behaviour '
              'of the original language model?', 19, w=1500, bold=True)
text(800, 66, 'A sparse autoencoder is trained on activations of a frozen LLM; its reconstruction is '
              'spliced back into the same frozen model and the two pathways are compared.', 12.5, w=1500)

CY = 390                                   # main-row centre line
rect(30, CY - 60, 140, 120, [('Input tokens', 14, True), (r'$t_1,\ldots,t_T$', 14, False)])
arrow([(170, CY), (200, CY)])
rect(200, CY - 80, 170, 160, [('Frozen LLM', 14, True), ('GPT-2 Small', 12, False), ('Gemma-2-2B', 12, False),
                              (r'layers $1,\ldots,\ell-1$', 12, False)])
arrow([(370, CY), (400, CY)])
rect(400, CY - 45, 120, 90, [(r'$\mathbf{x}\in\mathbb{R}^{d}$', 16, False), ('original activation', 11, False)])
text(460, CY + 62, r'layer-$\ell$ residual stream', 10.5, w=170)
arrow([(520, CY), (585, CY)], lw=1.6)

# original pathway (top lane)
line([(460, CY - 45), (460, 160)], lw=1.4)
arrow([(460, 160), (1395, 160)], lw=1.4)
text(930, 146, r'Original pathway: $\mathbf{x}$ continues unchanged', 12.5, w=600)
rect(1395, 110, 175, 100, [('Frozen LLM', 14, True), (r'layers $\ell,\ldots,L$', 12, False),
                           (r'output $p(\cdot\mid t)$', 13, False)])

# ---------------- SAE (focus)
SX, SY, SW, SH = 540, 240, 660, 510
rect(SX, SY, SW, SH, lw=2.4)
text(870, 268, 'Sparse autoencoder', 17, w=400, bold=True)
text(870, 290, 'trained on frozen activations (ReLU · TopK · JumpReLU)', 11.5, w=500)

poly([(585, 350), (755, 305), (755, 475), (585, 430)], [('Encoder', 14, True), (r'$d\rightarrow m$', 12, False)],
     'north')
arrow([(755, CY), (852, CY)], lw=1.6)
LX, LW, LH, NL = 854, 32, 14.5, 12
active = {1, 5, 8}
for i in range(NL):
    E.append(dict(k='rect', x=LX, y=305 + i * LH, w=LW, h=LH, lines=[], lw=1.0,
                  fill=FG if i in active else BG, dy=None))
arrow([(LX + LW, CY), (983, CY)], lw=1.6)
poly([(985, 305), (1155, 350), (1155, 430), (985, 475)], [('Decoder', 14, True), (r'$m\rightarrow d$', 12, False)],
     'south')

text(670, 512, r'$\mathbf{h}=W_{\mathrm{enc}}(\mathbf{x}-\mathbf{b}_{\mathrm{dec}})+\mathbf{b}_{\mathrm{enc}}$', 13.5,
     w=230)
text(870, 497, 'sparse latents', 11, w=160)
text(870, 522, r'$\mathbf{z}=f(\mathbf{h})\in\mathbb{R}^{m}$', 13.5, w=180)
text(870, 547, r'$\Vert\mathbf{z}\Vert_0\ll m$', 13.5, w=180)
text(1070, 512, r'$\hat{\mathbf{x}}=W_{\mathrm{dec}}\,\mathbf{z}+\mathbf{b}_{\mathrm{dec}}$', 13.5, w=230)
text(1070, 537, r'$\Vert W_{\mathrm{dec},:i}\Vert_2=1$', 12, w=230)

line([(SX + 25, 572), (SX + SW - 25, 572)], lw=0.8)
text(870, 590, r'Sparsity function $f$ and penalty $P$ (one architecture, three variants)', 12, w=600)
for cx, name, f, p in [
        (650, 'ReLU', r'$f(\mathbf{h})=\max(0,\mathbf{h})$', r'$P=\Vert\mathbf{z}\Vert_1$  ($\mathbf{z}\geq 0$)'),
        (860, 'TopK', r'$f(\mathbf{h})=\mathrm{TopK}_k(\max(0,\mathbf{h}))$', r'$P=0$  ($k$ fixed)'),
        (1077, 'JumpReLU', r'$f(\mathbf{h})=\max(0,\mathbf{h})\odot H(\mathbf{h}-\theta)$',
         r'$P=\Sigma_i\,s\left((h_i-\theta_i)/\tau\right)$')]:
    text(cx, 616, name, 13.5, w=220, bold=True)
    text(cx, 641, f, 12.5, w=230)
    text(cx, 666, p, 12.5, w=230)
line([(SX + 25, 684), (SX + SW - 25, 684)], lw=0.8)
text(870, 708, r'$\mathcal{L}=\frac{1}{d}\Vert\mathbf{x}-\hat{\mathbf{x}}\Vert_2^2+\lambda\,P(\mathbf{z})$', 15.5,
     w=320)
text(870, 735, r'$\lambda$ adapted online to the target $L_0$; $H$: step, $s$: logistic sigmoid with temperature '
               r'$\tau$ (straight-through)', 10.5, w=640)

# ---------------- reconstructed pathway
arrow([(1155, CY), (1240, CY)], lw=1.6)
rect(1240, CY - 45, 120, 90, [(r'$\hat{\mathbf{x}}\in\mathbb{R}^{d}$', 16, False), ('reconstruction', 11, False)])
arrow([(1360, CY), (1395, CY)], lw=1.6)
rect(1395, CY - 80, 175, 160, [('Frozen LLM', 14, True), (r'layers $\ell,\ldots,L$', 12, False),
                               ('(same weights)', 11, False), (r'output $\hat{p}(\cdot\mid t)$', 13, False)])
text(1405, CY + 105, 'Reconstructed pathway:', 12.5, w=330)
text(1405, CY + 127, r'splice $\mathbf{x}\leftarrow\hat{\mathbf{x}}$ at layer $\ell$', 12.5, w=330)

# ---------------- four faithfulness dimensions
text(800, 786, r'Faithfulness: original ($\mathbf{x}$, $p$) vs reconstructed ($\hat{\mathbf{x}}$, $\hat{p}$) pathway',
     15, w=1200, bold=True)
BX, BW, BG_, BY, BH = 30, 370, 20, 805, 170
dims = [
    ('1  Reconstruction', [r'$\mathrm{MSE}=\frac{1}{d}\Vert\mathbf{x}-\hat{\mathbf{x}}\Vert_2^2$,   '
                           r'$\cos(\mathbf{x},\hat{\mathbf{x}})$',
                           r'$\mathrm{EV}=1-\sum\Vert\mathbf{x}-\hat{\mathbf{x}}\Vert_2^2\,/\,'
                           r'\sum\Vert\mathbf{x}-\bar{\mathbf{x}}\Vert_2^2$',
                           r'$L_0=\Vert\mathbf{z}\Vert_0$, matched across architectures'],
     r'MSE $\downarrow$   EV $\uparrow$   cos $\uparrow$   at matched $L_0$'),
    ('2  Information preservation', [r'frozen probe $\mathbf{w}$ fitted on $\mathbf{x}$ only',
                                     r'$\Delta\mathrm{AUROC}=\mathrm{AUROC}(\mathbf{w}^{\top}\mathbf{x})'
                                     r'-\mathrm{AUROC}(\mathbf{w}^{\top}\hat{\mathbf{x}})$',
                                     'truth (Geometry of Truth), honesty (RepE)'],
     r'$\Delta\mathrm{AUROC}\rightarrow 0$'),
    ('3  Behavioural preservation', [r'$p=p(\cdot\mid t)$,   $\hat{p}=p(\cdot\mid t;\ \mathbf{x}\leftarrow'
                                     r'\hat{\mathbf{x}})$',
                                     r'$\mathrm{KL}(p\,\Vert\,\hat{p})$',
                                     r'$\Delta\mathrm{CE}=\mathrm{CE}(\hat{p})-\mathrm{CE}(p)$'],
     r'$\mathrm{KL}\rightarrow 0$   $\Delta\mathrm{CE}\rightarrow 0$'),
    ('4  Causal preservation', [r'$\Delta=\mathbf{x}_A-\mathbf{x}_B$,   '
                                r'$\hat{\Delta}=\hat{\mathbf{x}}_A-\hat{\mathbf{x}}_B$',
                                r'$T(\mathbf{v})=\mathrm{KL}(p_A\Vert p_B)-\mathrm{KL}(p_A\Vert p_{B+\mathbf{v}})$',
                                r'patch at layer $\ell$, against matched random controls'],
     r'$T(\hat{\Delta})\,/\,T(\Delta)\rightarrow 1$'),
]
for j, (title, body, target) in enumerate(dims):
    x0 = BX + j * (BW + BG_)
    rect(x0, BY, BW, BH)
    cx = x0 + BW / 2
    text(cx, BY + 26, title, 14.5, w=BW - 20, bold=True)
    for i, s in enumerate(body):
        text(cx, BY + 60 + i * 28, s, 12.5, w=BW - 16)
    line([(x0 + 20, BY + 136), (x0 + BW - 20, BY + 136)], lw=0.6)
    text(cx, BY + 154, target, 12.5, w=BW - 16)

text(800, 996, 'GPT-2 Small: in-house ReLU, TopK and JumpReLU SAEs (d = 768, m = 3072) at blocks.ℓ.hook_resid_pre. '
               'Gemma-2-2B: public Gemma Scope JumpReLU SAEs (d = 2304, m = 16,384). '
               'BOS is excluded from SAE training and splicing.', 10.5, w=1540)


# ----------------------------------------------------------------- matplotlib renderer (SVG + PNG)
def render_mpl(out):
    plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'svg.fonttype': 'path',
                         'svg.hashsalt': NAME})   # deterministic SVG ids
    fig = plt.figure(figsize=(W / 100, H / 100), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_facecolor(BG)
    ax.axis('off')
    for e in E:
        if e['k'] == 'rect':
            ax.add_patch(Rectangle((e['x'], e['y']), e['w'], e['h'], fc=e['fill'], ec=FG, lw=e['lw']))
            n = len(e['lines'])
            if n:
                sizes = [s for _, s, _ in e['lines']]
                gaps = [max(a, b) * 1.75 for a, b in zip(sizes, sizes[1:])]
                y = e['y'] + e['h'] / 2 - sum(gaps) / 2
                for i, (s, sz, b) in enumerate(e['lines']):
                    ax.text(e['x'] + e['w'] / 2, y, s, color=FG, fontsize=sz, ha='center', va='center',
                            fontweight='bold' if b else 'normal')
                    if i < n - 1:
                        y += gaps[i]
        elif e['k'] == 'poly':
            ax.add_patch(Polygon(e['pts'], closed=True, fc=BG, ec=FG, lw=e['lw']))
            cx = sum(p[0] for p in e['pts']) / 4
            cy = sum(p[1] for p in e['pts']) / 4
            for i, (s, sz, b) in enumerate(e['label']):
                ax.text(cx, cy - 12 + i * 26, s, color=FG, fontsize=sz, ha='center', va='center',
                        fontweight='bold' if b else 'normal')
        elif e['k'] == 'text':
            ax.text(e['x'], e['y'], e['s'], color=FG, fontsize=e['size'], ha=e['ha'], va='center',
                    fontweight='bold' if e['bold'] else 'normal')
        elif e['k'] == 'arrow':
            pts = e['pts']
            for a, b in zip(pts[:-2], pts[1:-1]):
                ax.plot([a[0], b[0]], [a[1], b[1]], color=FG, lw=e['lw'], solid_capstyle='butt')
            style = '-|>' if e['head'] else '-'
            ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle=style, mutation_scale=14, color=FG,
                                         lw=e['lw'], shrinkA=0, shrinkB=0))
    fig.savefig(out / f'{NAME}.svg', facecolor=BG, metadata={'Date': None})
    fig.savefig(out / f'{NAME}.png', dpi=400, facecolor=BG)
    plt.close(fig)


# ----------------------------------------------------------------- draw.io renderer
def tex(s):
    """$...$ -> MathJax inline \\(...\\); escape the rest as HTML."""
    parts = re.split(r'(\$[^$]*\$)', s)
    out = []
    for p in parts:
        if p.startswith('$') and p.endswith('$') and len(p) > 1:
            out.append(r'\(' + escape(p[1:-1]) + r'\)')
        else:
            out.append(escape(p).replace('  ', '&nbsp; '))
    return ''.join(out)


def html_lines(lines):
    return ''.join(f'<div style="font-size:{round(sz * PX)}px;{"font-weight:bold;" if b else ""}">{tex(s)}</div>'
                   for s, sz, b in lines)


def render_drawio(out):
    base = (f'fontFamily={FONT};fontColor={FG};strokeColor={FG};html=1;whiteSpace=wrap;')
    cells, n = [], [1]

    def nid():
        n[0] += 1
        return f'c{n[0]}'

    for e in E:
        if e['k'] == 'rect':
            st = base + f'rounded=0;fillColor={e["fill"]};strokeWidth={e["lw"]};verticalAlign=middle;align=center;'
            cells.append((nid(), html_lines(e['lines']), st, e['x'], e['y'], e['w'], e['h'], None))
        elif e['k'] == 'poly':
            xs = [p[0] for p in e['pts']]
            ys = [p[1] for p in e['pts']]
            x, y, w, h = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
            # draw.io trapezoid: default narrow edge on top; north rotates it to the left, south to the right
            st = base + (f'shape=trapezoid;perimeter=trapezoidPerimeter;fixedSize=0;size={45 / 170:.3f};'
                         f'direction={e["direction"]};fillColor={BG};strokeWidth={e["lw"]};')
            cells.append((nid(), html_lines(e['label']), st, x, y, w, h, None))
        elif e['k'] == 'text':
            hgt = 46 if r'\frac' in e['s'] else 32
            st = base + (f'text;strokeColor=none;fillColor=none;align={e["ha"]};verticalAlign=middle;'
                         f'fontSize={round(e["size"] * PX)};{"fontStyle=1;" if e["bold"] else ""}')
            x = e['x'] - e['w'] / 2 if e['ha'] == 'center' else e['x']
            cells.append((nid(), tex(e['s']), st, x, e['y'] - hgt / 2, e['w'], hgt, None))
        elif e['k'] == 'arrow':
            st = (f'endArrow={"block" if e["head"] else "none"};endFill=1;html=1;rounded=0;strokeColor={FG};'
                  f'strokeWidth={e["lw"]};')
            cells.append((nid(), '', st, None, None, None, None, e['pts']))
    xml = [f'<mxfile host="build_sae_architecture.py" type="device">',
           f'<diagram id="sae-arch" name="SAE faithfulness architecture">',
           f'<mxGraphModel dx="{W}" dy="{H}" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" '
           f'fold="1" page="1" pageScale="1" pageWidth="{W}" pageHeight="{H}" math="1" shadow="0" '
           f'background="{BG}">', '<root>', '<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
    for cid, val, st, x, y, w, h, pts in cells:
        if pts is None:
            xml.append(f'<mxCell id="{cid}" value={quoteattr(val)} style={quoteattr(st)} vertex="1" parent="1">'
                       f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" as="geometry"/>'
                       f'</mxCell>')
        else:
            mids = ''.join(f'<mxPoint x="{p[0]}" y="{p[1]}"/>' for p in pts[1:-1])
            xml.append(f'<mxCell id="{cid}" value="" style={quoteattr(st)} edge="1" parent="1">'
                       f'<mxGeometry relative="1" as="geometry">'
                       f'<mxPoint x="{pts[0][0]}" y="{pts[0][1]}" as="sourcePoint"/>'
                       f'<mxPoint x="{pts[-1][0]}" y="{pts[-1][1]}" as="targetPoint"/>'
                       + (f'<Array as="points">{mids}</Array>' if mids else '') +
                       '</mxGeometry></mxCell>')
    xml += ['</root>', '</mxGraphModel>', '</diagram>', '</mxfile>']
    (out / f'{NAME}.drawio').write_text('\n'.join(xml), encoding='utf-8')


if __name__ == '__main__':
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    out.mkdir(parents=True, exist_ok=True)
    render_mpl(out)
    render_drawio(out)
    print('wrote', *sorted(p.name for p in out.iterdir()))
