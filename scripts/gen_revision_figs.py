# -*- coding: utf-8 -*-
"""gen_revision_figs.py — 修改稿全套图片重绘（v2数据，唯一数据源 source_code/reports/）
产出：fig3/fig4/fig5/fig6/fig7/fig8 + supplementary 图件 + tab_ablation_detail.tex

用法（在 source_code 下）：
  & Python311 scripts\\gen_revision_figs.py                 # 全量（默认跳过 fig5）
  & Python311 scripts\\gen_revision_figs.py --only fig4 fig6  # 只重生成指定图
  & Python311 scripts\\gen_revision_figs.py --all           # 含 fig5（正文已删，默认跳过）

设计规范：
- 单栏图（\columnwidth ≈ 3.5 in）：fig3 / fig4 / fig6 / fig8 / supp fig2
- 全宽图（\textwidth ≈ 7.0–7.16 in）：fig7 / supp fig1 / fig6 / figS
- 统一 Times serif + STIX 数学字体，8pt 基准字号，pdf.fonttype=42 矢量输出
- 数据零改动：仅呈现（布局/字号/标注/配色）
"""
import argparse
import io
import json
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

ROOT = Path(__file__).parent.parent          # source_code
PROJ = ROOT.parent                           # 项目根
REP = ROOT / 'reports'
FIG = PROJ / 'figures'
SUPP = FIG / 'supp'
FIG.mkdir(exist_ok=True)
SUPP.mkdir(exist_ok=True)

# ── 统一风格（与正文 Times 正文匹配；STIX 数学字体与 Times 协调）───────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5,
    'axes.linewidth': 0.7, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
    'xtick.major.size': 2.2, 'ytick.major.size': 2.2,
    'xtick.major.pad': 1.5, 'ytick.major.pad': 1.5,
    'legend.framealpha': 0.92, 'legend.edgecolor': '0.75',
})

INSTANCES = [f'I{i}' for i in range(1, 11)]
BASE = ['nsga2_topsis', 'nsga2_vikor', 'moead_topsis', 'spea2_topsis',
        'commea_topsis', 'hdmmode_topsis', 'mmoeadc_topsis']
ORDER8 = ['rcmd_v2'] + BASE
LBL = {'rcmd_v2': 'RCMD-EA', 'nsga2_topsis': 'NSGA-II+T', 'nsga2_vikor': 'NSGA-II+V',
       'moead_topsis': 'MOEA/D+T', 'spea2_topsis': 'SPEA2+T', 'commea_topsis': 'CoMMEA+T',
       'hdmmode_topsis': 'HDMMODE+T', 'mmoeadc_topsis': 'MMOEADC+T',
       'abv2_no_rcmd': '-C1', 'abv2_no_explore': '-C2', 'abv2_fixed_pref': '-C3',
       'abv2_no_dual_archive': '-DA'}
COL = {'rcmd_v2': '#d62728', 'nsga2_topsis': '#1f77b4', 'nsga2_vikor': '#2ca02c',
       'moead_topsis': '#9467bd', 'spea2_topsis': '#8c564b', 'commea_topsis': '#e377c2',
       'hdmmode_topsis': '#7f7f7f', 'mmoeadc_topsis': '#bcbd22',
       'abv2_no_rcmd': '#ff7f0e', 'abv2_no_explore': '#17becf', 'abv2_fixed_pref': '#6a3d9a',
       'abv2_no_dual_archive': '#949494'}
MK = {'rcmd_v2': 'o', 'nsga2_topsis': 's', 'nsga2_vikor': '^', 'moead_topsis': 'D',
      'spea2_topsis': 'v', 'commea_topsis': 'P', 'hdmmode_topsis': 'X', 'mmoeadc_topsis': '*',
      'abv2_no_rcmd': 'o', 'abv2_no_explore': 's', 'abv2_fixed_pref': '^',
      'abv2_no_dual_archive': 'v'}
# 消融变体线型（-DA 用灰点线，与 SI 图注 "grey dotted" 一致）
LSTYLE = {'rcmd_v2': '-', 'abv2_no_rcmd': '-', 'abv2_no_explore': '-',
          'abv2_fixed_pref': '--', 'abv2_no_dual_archive': ':'}

C_BLUE = '#1565c0'     # w_g / phase 1
C_ORANGE = '#e65100'   # δ_g / exploitation
C_RED = '#d62728'      # phase 8 / RCMD
C_GREEN = '#2e7d32'    # φ_t
HIGHLIGHT = '#0173b2'  # RCMD-EA 行高亮


# ══════════════════════════════════════════════════════════════════════════
# 数据加载（唯一数据源 source_code/reports/，懒加载 + 缓存）
# ══════════════════════════════════════════════════════════════════════════
def load(algo, inst):
    out = []
    for run in range(30):
        p = REP / algo / f'{inst}_run{run:03d}.json'
        d = json.loads(p.read_text())
        ph = d['phases']
        out.append((np.mean([x['best_rmi'] for x in ph]),
                    np.nanmean([x.get('hd', np.nan) for x in ph])))
    return np.array(out)


def load3(algo, inst):
    out = []
    for run in range(30):
        d = json.loads((REP / algo / f'{inst}_run{run:03d}.json').read_text())
        ph = d['phases']
        out.append((np.mean([x['best_rmi'] for x in ph]),
                    np.nanmean([x.get('hd', np.nan) for x in ph]),
                    float(np.sum([x.get('hv', 0.0) for x in ph])) / len(ph)))
    return np.array(out)


_D = None
_D3 = None


def get_D():
    global _D
    if _D is None:
        algos = ORDER8 + ['abv2_no_rcmd', 'abv2_no_explore', 'abv2_fixed_pref',
                          'abv2_no_dual_archive']
        _D = {a: {i: load(a, i) for i in INSTANCES} for a in algos}
        print('data D loaded')
    return _D


def get_D3():
    global _D3
    if _D3 is None:
        ABLS = ['rcmd_v2', 'abv2_no_rcmd', 'abv2_no_explore', 'abv2_fixed_pref']
        _D3 = {a: {i: load3(a, i) for i in INSTANCES} for a in ABLS}
        print('data D3 loaded')
    return _D3


def load_signal():
    return json.loads((PROJ / 'revision' / 'fig5_signal_data.json').read_text())


def savefig(fig, path):
    fig.savefig(path, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'saved {path.name}')


# ══════════════════════════════════════════════════════════════════════════
# fig4: Friedman 排名热图（正文 Fig.3，单栏 3.5in，左右并排，总高 ≤2.2in）
#   - 行=8算法，列=I1-I10；每张热图右侧独立 Mean 窄条（白底加粗）
#   - 底部共享 colorbar；RCMD-EA 行蓝色描边
# ══════════════════════════════════════════════════════════════════════════
def fig4_ranking():
    from scipy import stats as sps
    D = get_D()
    mats = {}
    for mi in (0, 1):
        mat = np.array([[D[a][i][:, mi].mean() for a in ORDER8] for i in INSTANCES])
        mats[mi] = np.array([sps.rankdata(-row) for row in mat])  # (10 inst, 8 algos)

    FIG_W, FIG_H = 3.5, 2.0
    # 几何（英寸）：y标签栏 | 热图(10列)+Mean窄条 | 间隙 | 热图2+Mean窄条
    X0 = 0.56
    CELL_W, NCOL, NROW = 0.121, 10, 8
    HEAT_W = CELL_W * NCOL
    MEAN_W = 0.16
    GAP_HM = 0.03                   # 热图 -> Mean 窄条
    GAP_P = 0.10                    # 两个面板之间
    CELL_H = 0.158
    HEAT_H = CELL_H * NROW
    T1_TOP = FIG_H - 0.18
    BOT = T1_TOP - HEAT_H
    CB_Y, CB_H = 0.10, 0.065

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    def frac(x_in, y_in, w_in, h_in):
        return [x_in / FIG_W, y_in / FIG_H, w_in / FIG_W, h_in / FIG_H]

    xs = [X0, X0 + HEAT_W + GAP_HM + MEAN_W + GAP_P]
    im = None
    for k, (mi, title) in enumerate([(0, 'RMI rank'), (1, 'HD rank')]):
        ranks = mats[mi].T                                   # (8 algos, 10 inst)
        ax = fig.add_axes(frac(xs[k], BOT, HEAT_W, HEAT_H))
        im = ax.imshow(ranks, cmap='RdYlGn', vmin=1, vmax=8, aspect='auto',
                       interpolation='nearest')
        ax.set_title(title, loc='left', fontsize=7.5, fontweight='bold', pad=2.5)
        ax.set_xticks(range(NCOL))
        ax.set_xticklabels(INSTANCES, fontsize=5.2)
        ax.set_yticks(range(NROW))
        ax.set_yticklabels([LBL[a] for a in ORDER8] if k == 0 else [], fontsize=5.8)
        ax.tick_params(length=0, pad=1.0)
        ax.grid(False)
        ax.set_xticks(np.arange(-0.5, NCOL, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, NROW, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=0.6)
        ax.tick_params(which='minor', length=0)
        for r in range(NROW):
            for c in range(NCOL):
                v = ranks[r, c]
                rgba = plt.cm.RdYlGn((v - 1) / 7)
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(c, r, f'{v:g}', ha='center', va='center', fontsize=5.2,
                        color='white' if lum < 0.5 else '#1a1a1a')
        ax.add_patch(Rectangle((-0.5, -0.5), NCOL, 1, fill=False,
                               edgecolor=HIGHLIGHT, linewidth=1.1, zorder=5))
        if k == 0:
            ax.get_yticklabels()[0].set_fontweight('bold')
        # Mean 独立窄条（白底加粗，与热图 colormap 视觉分离）
        axm = fig.add_axes(frac(xs[k] + HEAT_W + GAP_HM, BOT, MEAN_W, HEAT_H))
        axm.set_xlim(0, 1); axm.set_ylim(NROW - 0.5, -0.5)
        axm.axis('off')
        axm.add_patch(Rectangle((0, -0.5), 1, NROW, facecolor='#f4f4f4',
                                edgecolor='#9a9a9a', linewidth=0.6, zorder=1))
        means = ranks.mean(axis=1)
        for r in range(NROW):
            axm.text(0.5, r, f'{means[r]:.1f}', ha='center', va='center',
                     fontsize=5.4, fontweight='bold', color='#1a1a1a', zorder=3)
        axm.text(0.5, NROW - 0.5 + 0.62, 'Mean', ha='center', va='top',
                 fontsize=5.8, fontweight='bold')
        axm.add_patch(Rectangle((0, -0.5), 1, 1, fill=False,
                                edgecolor=HIGHLIGHT, linewidth=1.1, zorder=5))

    cb_ax = fig.add_axes(frac(X0, CB_Y, xs[1] + HEAT_W + GAP_HM + MEAN_W - X0, CB_H))
    cb = fig.colorbar(im, cax=cb_ax, orientation='horizontal')
    cb.set_ticks(range(1, 9))
    cb.ax.tick_params(labelsize=5.5, length=2, width=0.6, pad=1.5)
    cb.outline.set_linewidth(0.6)
    savefig(fig, FIG / 'fig4_ranking_summary.pdf')


# ══════════════════════════════════════════════════════════════════════════
# fig5: RMI vs 规模（正文已删除该图；默认跳过不重生成，保留现有文件）
# ══════════════════════════════════════════════════════════════════════════
def fig5_convergence():
    D = get_D()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    x = np.arange(10)
    for a in ORDER8:
        m = np.array([D[a][i][:, 0].mean() for i in INSTANCES])
        s = np.array([D[a][i][:, 0].std() for i in INSTANCES])
        ax.plot(x, m, marker=MK[a], color=COL[a], label=LBL[a], markersize=3.5, linewidth=1.2)
        if a == 'rcmd_v2':
            ax.fill_between(x, m - s, m + s, color=COL[a], alpha=0.15)
    ax.set_xticks(x); ax.set_xticklabels(INSTANCES)
    ax.set_xlabel('Instance'); ax.set_ylabel('Mean RMI (30 runs)')
    ax.legend(ncol=2, fontsize=6, loc='lower left')
    fig.tight_layout(); fig.savefig(FIG / 'fig5_convergence.pdf', bbox_inches='tight'); plt.close(fig)
    print('fig5 done')


# ══════════════════════════════════════════════════════════════════════════
# fig3: RCMD 概念图（正文 Fig.2，单栏 3.5in，左(a)右(b)与正文图注一致）
#   (a) 模态簇散点：'R1 filter' 标注移到空白区并加引出线，图例避让数据
#   (b) w_g/δ_g 双轴：淡色背景带区分两阶段 + 顶部小字，曲线直接标注替代图例
# ══════════════════════════════════════════════════════════════════════════
def fig3_concept():
    sig = load_signal()
    rng = np.random.default_rng(7)
    fig = plt.figure(figsize=(3.5, 2.65))

    # ── (a) 模态簇 ──
    ax = fig.add_axes([0.115, 0.15, 0.42, 0.76])
    ax.set_aspect('equal', adjustable='box')
    c1 = rng.normal([1.5, 1.5], 0.32, (26, 2))
    c2 = rng.normal([4.2, 3.4], 0.36, (22, 2))
    ax.scatter(*c1.T, s=9, c='#1f77b4', edgecolors='k', linewidths=0.35, label='feasible, mode 1')
    ax.scatter(*c2.T, s=9, c='#2ca02c', edgecolors='k', linewidths=0.35, label='feasible, mode 2')
    inf = np.array([[2.6, 2.4], [3.2, 2.0], [2.9, 3.1], [3.6, 2.6], [2.3, 2.8], [3.0, 2.5]])
    ax.scatter(*inf.T, s=20, marker='x', c=C_RED, linewidths=1.1, label='infeasible ($g>0$)')
    for ctr, r in [([1.5, 1.5], 0.85), ([4.2, 3.4], 0.95)]:
        ax.add_patch(plt.Circle(ctr, r, fill=False, ls='--', ec='#555555', lw=0.8))
    # R1 filter 标注：移到右下空白区 + 引出线指向不可行解簇
    ax.annotate('R1 filter\n(MORI)', xy=(2.95, 2.35), xytext=(3.55, 1.05),
                fontsize=5.8, color=C_RED, ha='left', va='center',
                arrowprops=dict(arrowstyle='->', color=C_RED, lw=0.8,
                                shrinkA=2, shrinkB=3))
    ax.set_xlim(0.4, 5.8); ax.set_ylim(0.3, 4.9)
    ax.set_xticks([1, 3, 5]); ax.set_yticks([1, 3])
    ax.set_xlabel('decision dimension 1', fontsize=6.5, labelpad=1.5)
    ax.set_ylabel('decision dimension 2', fontsize=6.5, labelpad=1.5)
    ax.tick_params(labelsize=6)
    ax.set_title('(a) Modal clusters', fontsize=7.5, pad=3)
    ax.legend(fontsize=5.2, loc='upper left', borderpad=0.35, labelspacing=0.3,
              handlelength=0.9, handletextpad=0.35)
    ax.grid(False)

    # ── (b) 自适应半径与信号（I7 phase 1 实测）──
    ph1 = sig['phases'][0]
    wg = ph1['w_g_history']; dg = ph1['delta_g_history']
    n1 = len(wg)
    Gexp = 40   # explore_ratio=0.4 × G_max=100（rcmd_ea.py L552），分界与实测一致
    ax = fig.add_axes([0.655, 0.19, 0.23, 0.66])
    gen = np.arange(1, n1 + 1)
    # 两阶段淡色背景带
    ax.axvspan(0.0, Gexp + 0.5, color='#dbeafe', alpha=0.55, zorder=0)
    ax.axvspan(Gexp + 0.5, n1 + 0.5, color='#ffe8cc', alpha=0.55, zorder=0)
    ax.axvline(Gexp + 0.5, color='#888888', lw=0.7, ls=':', zorder=1)
    ax.plot(gen, wg, '-', color=C_BLUE, lw=1.3, zorder=3)
    ax2 = ax.twinx()
    ax2.plot(gen, dg, '--', color=C_ORANGE, lw=1.3, zorder=3)
    # 顶部小字阶段标注（数据区之外；exploration/exploitation 由图注说明）
    ax.text((1 + Gexp) / 2, 0.525, 'Phase I', ha='center', va='top',
            fontsize=5.6, color=C_BLUE)
    ax.text((Gexp + n1) / 2 + 1, 0.525, 'Phase II', ha='center', va='top',
            fontsize=5.6, color='#8a5a00')
    # 曲线末端直接标注（替代图例，白底避让）
    ax.text(47.5, 0.295, '$w_g$', color=C_BLUE, fontsize=6.5, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.85))
    ax2.text(47.5, 1.705, '$\\delta_g$', color=C_ORANGE, fontsize=6.5, va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.85))
    ax.set_xlim(0, 56); ax.set_ylim(0, 0.55)
    ax2.set_ylim(1.55, 2.20)
    ax.set_xticks([1, 10, 20, 30, 40, 50])
    ax.set_yticks([0, 0.25, 0.5])
    ax2.set_yticks([1.6, 1.8, 2.0, 2.2])
    ax.tick_params(labelsize=6); ax2.tick_params(labelsize=6)
    ax.set_xlabel('Generation', fontsize=6.5, labelpad=1.5)
    ax.set_ylabel('$w_g$', color=C_BLUE, fontsize=7)
    ax2.set_ylabel('$\\delta_g$', color=C_ORANGE, fontsize=7)
    ax.set_title('(b) Radius & signal\n(I7, phase 1)', fontsize=7.5, pad=3)
    ax.grid(False); ax2.grid(False)
    savefig(fig, FIG / 'fig3_rcmd_concept.pdf')


# ══════════════════════════════════════════════════════════════════════════
# fig6: 双信号（正文 Fig.5，单栏 3.5in，左右并排，总高 ≤1.6in）
#   (a) φ_t 柱状图 + 近均匀威胁区注释带 + 均值线（实测值不变）
#   (b) w_g：phase 1 / phase 8 曲线 + 代际切换点标注（gen 42, 0.1→0.5）
# ══════════════════════════════════════════════════════════════════════════
def fig6_dual_signal():
    sig = load_signal()
    fig = plt.figure(figsize=(3.5, 1.55))

    # ── (a) φ_t 柱状图（左）──
    ax = fig.add_axes([0.135, 0.245, 0.355, 0.565])
    phis = np.array(sig['phi_t'])
    x = np.arange(1, 11)
    ax.axhspan(phis.min(), phis.max(), color='#666666', alpha=0.10, zorder=0)
    ax.bar(x, phis, width=0.62, color=C_GREEN, edgecolor='#1b5e20', linewidth=0.4, zorder=2)
    mean_phi = phis.mean()
    ax.axhline(mean_phi, color='#1b5e20', lw=0.8, ls='--', zorder=3)
    ax.text(10.55, 0.00845, f'mean = {mean_phi:.4f}', ha='right', va='bottom',
            fontsize=5, color='#1b5e20')
    # 注释带说明（右上空白区）
    ax.text(0.98, 0.95, 'near-uniform threat regime\n' + r'$\phi_t \in [0.006,\,0.011]$',
            transform=ax.transAxes, ha='right', va='top', fontsize=5, color='#444444',
            linespacing=1.5)
    ax.set_xlim(0.35, 10.65); ax.set_ylim(0, 0.0135)
    ax.set_xticks(x)
    ax.set_yticks([0, 0.004, 0.008, 0.012])
    ax.set_yticklabels(['0.000', '0.004', '0.008', '0.012'])
    ax.tick_params(labelsize=5.5, pad=1.2)
    ax.set_xlabel('Phase', fontsize=6, labelpad=1.2)
    ax.set_ylabel(r'$\phi_t$', fontsize=7)
    ax.set_title(r'(a) Situational signal $\phi_t$', fontsize=6.5, pad=2)

    # ── (b) w_g 双相位（右）──
    ax = fig.add_axes([0.635, 0.245, 0.335, 0.565])
    w1 = sig['phases'][0]['w_g_history']
    w8 = sig['phases'][7]['w_g_history']
    g1 = np.arange(1, len(w1) + 1)
    g8 = np.arange(1, len(w8) + 1)
    ax.plot(g1, w1, '-', color=C_BLUE, lw=1.0, marker='o', markersize=1.5,
            markevery=5, label='phase 1 (exploration)', zorder=3)
    ax.plot(g8, w8, '-', color=C_RED, lw=1.0, label='phase 8 (refinement)', zorder=3)
    # 代际切换点：phase 8 的 w_g 在 gen 42 由 0.1 → 0.5（实测）
    sw = next(i for i in range(1, len(w8)) if w8[i] != w8[i - 1]) + 1
    ax.axvline(sw, color='#888888', lw=0.7, ls=':', zorder=2)
    ax.annotate(f'gen {sw}: explore $\\to$ refine',
                xy=(sw, 0.5), xytext=(sw + 3, 0.60), fontsize=5, color='#444444',
                va='top',
                arrowprops=dict(arrowstyle='->', color='#888888', lw=0.6,
                                shrinkA=1, shrinkB=2))
    ax.text(0.985, 0.80, '$w_g=0.5$', transform=ax.transAxes, ha='right', va='top',
            fontsize=5, color='#b71c1c')
    ax.set_xlim(0, 145); ax.set_ylim(0, 0.62)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 0.2, 0.4])
    ax.tick_params(labelsize=5.5, pad=1.2)
    ax.set_xlabel('Generation', fontsize=6, labelpad=1.2)
    ax.set_ylabel('$w_g$', fontsize=7)
    ax.set_title('(b) Search signal $w_g$', fontsize=6.5, pad=2)
    ax.legend(fontsize=5, loc='lower right', borderpad=0.3, handlelength=1.4,
              handletextpad=0.4, labelspacing=0.25)
    savefig(fig, FIG / 'fig6_dual_signal.pdf')


# ══════════════════════════════════════════════════════════════════════════
# fig7: 赋值热图（正文 Fig.6，全宽 7.0in）
#   - 威胁条带改为与主轴严格同 x 范围的 inset（贴附主轴上沿），彻底对齐
# ══════════════════════════════════════════════════════════════════════════
def fig7_assignment():
    asg = json.loads((PROJ / 'revision' / 'assignment_results_v2.json').read_text())
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8))
    fig.subplots_adjust(left=0.065, right=0.99, top=0.88, bottom=0.075,
                        wspace=0.14, hspace=0.45)
    for col, name in enumerate(['I2', 'I3']):
        e = asg[name]
        threats = np.array(e['threats_p2'])
        th_norm = threats / threats.max()
        for row, tag in enumerate(['rcmd_ea', 'nsga2_topsis']):
            ax = axes[row, col]
            shots = np.array(e[tag]['shots'])
            m, n_t = shots.shape
            vmax = max(shots.max(), 1)
            # 全部单元格浅底 + 浅灰描边（表格感），非零格叠加红色
            for i in range(m):
                for j in range(n_t):
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           facecolor='#f5f5f5', edgecolor='#d9d9d9',
                                           linewidth=0.4, zorder=1))
                    if shots[i, j] > 0:
                        frac = 0.25 + 0.6 * shots[i, j] / vmax
                        rgba = plt.cm.Reds(frac)
                        lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                               facecolor=rgba, edgecolor='white',
                                               linewidth=0.8, zorder=2))
                        ax.text(j, i, str(int(shots[i, j])), ha='center', va='center',
                                fontsize=6.5, fontweight='bold', zorder=3,
                                color='white' if lum < 0.5 else '#1a1a1a')
            ax.set_xlim(-0.5, n_t - 0.5)
            ax.set_ylim(m - 0.5, -0.5)
            ax.set_xticks(range(n_t))
            ax.set_xticklabels(range(1, n_t + 1), fontsize=6)
            ax.set_yticks(range(m))
            ax.set_yticklabels([f'W{i+1}' for i in range(m)], fontsize=6.5)
            ax.tick_params(length=0, pad=2)
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_linewidth(0.6)
            # 威胁条带：inset 贴附主轴上沿，xlim 与主轴完全一致 → 严格对齐
            strip_h = 0.16
            ax2 = ax.inset_axes([0, 1.0, 1, strip_h])
            ax2.set_xlim(-0.5, n_t - 0.5)
            ax2.set_ylim(0, 1)
            ax2.bar(range(n_t), th_norm, width=0.88,
                    color=[plt.cm.OrRd(0.3 + 0.6 * v) for v in th_norm],
                    edgecolor='white', linewidth=0.4, zorder=2)
            ax2.axis('off')
            ax2.add_patch(Rectangle((-0.5, 0), n_t, 1, fill=False,
                                    edgecolor='#bbbbbb', linewidth=0.6, zorder=3,
                                    clip_on=False))
            ttl = 'RCMD-EA' if tag == 'rcmd_ea' else 'NSGA-II+TOPSIS'
            pad_pt = ax.get_position().height * fig.get_figheight() * strip_h * 72 + 4
            ax.set_title(f'{name} — {ttl}  (RMI = {e[tag]["rmi"]:.2f})',
                         fontsize=7.5, pad=pad_pt)
            if row == 1:
                ax.set_xlabel('Target', fontsize=7, labelpad=2)
            if col == 0:
                ax.set_ylabel('Weapon', fontsize=7, labelpad=2)
    savefig(fig, FIG / 'fig7_assignment.pdf')


# ══════════════════════════════════════════════════════════════════════════
# fig8: 消融（正文 Fig.7，单栏 3.5in，三面板横向一行 + 顶部共享图例，总高 ≤2.0in）
#   x 轴用实例序号 1..10（xlabel=Instance），水平小字号不重叠
# ══════════════════════════════════════════════════════════════════════════
def fig8_ablation():
    D3 = get_D3()
    ABLS = ['rcmd_v2', 'abv2_no_rcmd', 'abv2_no_explore', 'abv2_fixed_pref']
    fig = plt.figure(figsize=(3.5, 2.0))
    x = np.arange(10)
    # 左边距 0.42in（ytick+ylabel），三面板等宽，间隙 0.12in
    left, pw, gap = 0.42 / 3.5, 0.923 / 3.5, 0.12 / 3.5
    panels = [(0, 'RMI'), (1, 'HD'), (2, 'HV (per-phase)')]
    handles = []
    for k, (metric, name) in enumerate(panels):
        ax = fig.add_axes([left + k * (pw + gap), 0.235, pw, 0.655])
        for a in ABLS:
            v = [D3[a][i][:, metric].mean() for i in INSTANCES]
            ln, = ax.plot(x, v, marker=MK[a], color=COL[a], label=LBL[a],
                          markersize=2.4, linewidth=1.0, linestyle=LSTYLE[a],
                          markeredgecolor='white', markeredgewidth=0.35)
            if k == 0:
                handles.append(ln)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i + 1) for i in range(10)], fontsize=5.5)
        ax.set_xlabel('Instance', fontsize=6.5, labelpad=1.2)
        ax.tick_params(labelsize=5.5, pad=1.2)
        ax.set_ylabel(name, fontsize=6.5)
        ax.grid(True, alpha=0.25, linewidth=0.5)
    fig.legend(handles, [LBL[a] for a in ABLS],
               ncol=4, fontsize=6, loc='upper center', bbox_to_anchor=(0.555, 0.995),
               columnspacing=0.9, handlelength=1.6, handletextpad=0.4,
               borderpad=0.35, frameon=False)
    savefig(fig, FIG / 'fig8_ablation.pdf')


# ══════════════════════════════════════════════════════════════════════════
# SI 图件（统一 Times serif + 按最终插入宽度设计）
# ══════════════════════════════════════════════════════════════════════════
def supp_fig1_main_comparison():
    D = get_D()
    fig, axes = plt.subplots(2, 10, figsize=(7.16, 3.3), sharey='row')
    for k, mi in enumerate([0, 1]):
        for j, i in enumerate(INSTANCES):
            ax = axes[k, j]
            data_l, labels, colors = [], [], []
            for a in ORDER8:
                data_l.append(D[a][i][:, mi]); labels.append(LBL[a]); colors.append(COL[a])
            bp = ax.boxplot(data_l, patch_artist=True, widths=0.7, showfliers=False,
                            medianprops=dict(color='k', lw=0.9),
                            boxprops=dict(lw=0.5), whiskerprops=dict(lw=0.5),
                            capprops=dict(lw=0.5))
            for patch, c in zip(bp['boxes'], colors):
                patch.set_facecolor(c); patch.set_alpha(0.75)
            if j == 0:
                ax.set_ylabel('RMI' if mi == 0 else 'HD', fontsize=7)
            ax.set_xticks(range(1, 9))
            ax.set_xticklabels([l.replace('+T', '') for l in labels], rotation=90,
                               fontsize=4.8)
            ax.set_title(i, fontsize=6.5, pad=2)
            ax.tick_params(length=1.5, pad=1)
            ax.grid(True, axis='y', alpha=0.2, linewidth=0.4)
    fig.tight_layout(pad=0.5)
    savefig(fig, SUPP / 'fig1_main_comparison.pdf')


def supp_fig2_rmi_heatmap():
    D = get_D()
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    adv = np.array([[np.median(D['rcmd_v2'][i][:, 0]) - np.median(D[a][i][:, 0])
                     for a in BASE] for i in INSTANCES])
    im = ax.imshow(adv, cmap='RdYlGn', vmin=-30, vmax=30, aspect='auto',
                   interpolation='nearest')
    for r in range(10):
        for c in range(7):
            rgba = plt.cm.RdYlGn((adv[r, c] + 30) / 60)
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(c, r, f'{adv[r, c]:.0f}', ha='center', va='center', fontsize=5.2,
                    color='white' if lum < 0.5 else '#1a1a1a')
    ax.set_xticks(range(7))
    ax.set_xticklabels([LBL[a].replace('+T', '') for a in BASE], rotation=90, fontsize=5.8)
    ax.set_yticks(range(10))
    ax.set_yticklabels(INSTANCES, fontsize=6)
    ax.tick_params(length=0, pad=1.5)
    ax.grid(False)
    ax.set_xticks(np.arange(-0.5, 7, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.6)
    ax.tick_params(which='minor', length=0)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.04)
    cb.ax.tick_params(labelsize=5.8, length=2, width=0.6)
    cb.set_label('RMI advantage\n(RCMD − baseline, median)', fontsize=6)
    cb.outline.set_linewidth(0.6)
    fig.tight_layout(pad=0.5)
    savefig(fig, SUPP / 'fig2_rmi_heatmap.pdf')


def supp_fig6_rmi_boxplot():
    D = get_D()
    SEL5 = ['rcmd_v2', 'nsga2_topsis', 'commea_topsis', 'hdmmode_topsis', 'mmoeadc_topsis']
    fig, axes = plt.subplots(2, 5, figsize=(7.16, 2.6), sharey=False)
    for j, i in enumerate(INSTANCES):
        ax = axes[j // 5, j % 5]
        data_l = [D[a][i][:, 0] for a in SEL5]
        bp = ax.boxplot(data_l, patch_artist=True, widths=0.6, showfliers=False,
                        medianprops=dict(color='k', lw=0.9),
                        boxprops=dict(lw=0.5), whiskerprops=dict(lw=0.5),
                        capprops=dict(lw=0.5))
        for patch, a in zip(bp['boxes'], SEL5):
            patch.set_facecolor(COL[a]); patch.set_alpha(0.75)
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels([LBL[a].replace('+T', '') for a in SEL5], rotation=90,
                           fontsize=5.2)
        ax.set_title(i, fontsize=6.5, pad=2)
        ax.tick_params(length=1.5, pad=1)
        ax.grid(True, axis='y', alpha=0.2, linewidth=0.4)
    fig.tight_layout(pad=0.5)
    savefig(fig, SUPP / 'fig6_rmi_boxplot.pdf')


def supp_figS_ablation_detail():
    D = get_D()
    ABLS5 = ['rcmd_v2', 'abv2_no_rcmd', 'abv2_no_explore', 'abv2_fixed_pref',
             'abv2_no_dual_archive']
    fig, axes = plt.subplots(2, 10, figsize=(7.16, 3.35), sharey='row')
    for k, mi in enumerate([0, 1]):
        for j, i in enumerate(INSTANCES):
            ax = axes[k, j]
            for a in ABLS5:
                v = D[a][i][:, mi]
                med = np.median(v)
                q1, q3 = np.percentile(v, 25), np.percentile(v, 75)
                ax.plot([j - 0.18, j + 0.18], [q1, q3], color=COL[a], lw=1.1,
                        linestyle=LSTYLE[a])
                ax.plot(j, med, marker=MK[a], color=COL[a], markersize=2.6,
                        linestyle='None', markeredgecolor='white', markeredgewidth=0.3)
            ax.set_xlim(-0.5, 0.5)
            ax.set_xticks([])
            ax.set_title(i, fontsize=6, pad=2)
            ax.tick_params(length=1.5, pad=1)
            ax.grid(True, axis='y', alpha=0.2, linewidth=0.4)
            if j == 0:
                ax.set_ylabel('RMI' if mi == 0 else 'HD', fontsize=7)
    handles = [Line2D([0], [0], color=COL[a], marker=MK[a], linestyle=LSTYLE[a],
                      label=LBL[a], markersize=3.5) for a in ABLS5]
    fig.legend(handles, [LBL[a] for a in ABLS5], ncol=5, fontsize=6.5,
               loc='upper center', bbox_to_anchor=(0.5, 1.0), columnspacing=1.2,
               handlelength=1.8, handletextpad=0.5, frameon=False)
    fig.tight_layout(pad=0.5, rect=[0, 0, 1, 0.93])
    savefig(fig, SUPP / 'figS_ablation_detail.pdf')


def supp_tab_ablation_detail():
    D = get_D()
    ABLS5 = ['rcmd_v2', 'abv2_no_rcmd', 'abv2_no_explore', 'abv2_fixed_pref',
             'abv2_no_dual_archive']
    lines = ['% Auto-generated by gen_revision_figs.py',
             '\\begin{table*}[htb]',
             '  \\caption{Ablation per-instance detail: mean $\\pm$ std of RMI and HD over 30 runs',
             '    (revised v2 ablation suite). $-$C1 replaces RCMD by plain Pareto dominance;',
             '    $-$C2 disables the exploration phase; $-$C3 fixes $\\alpha = 0.5$; $-$DA disables',
             '    the dual archive.}',
             '  \\label{tab:ablation_detail}',
             '  \\centering\\scriptsize',
             '  \\setlength{\\tabcolsep}{3pt}',
             '  \\resizebox{\\textwidth}{!}{%',
             '  \\begin{tabular}{@{}l' + 'c' * 10 + '@{}}',
             '    \\toprule']
    lines.append('    RMI & ' + ' & '.join(INSTANCES) + ' \\\\')
    lines.append('    \\midrule')

    def tm(s):
        return re.sub(r'(?<![\w$}])-(?=\d)', r'$-$', s)

    for a in ABLS5:
        cells = []
        for i in INSTANCES:
            v = D[a][i][:, 0]
            cells.append(tm(f'{v.mean():.2f}') + '$\\pm$' + f'{v.std():.2f}')
        lines.append(f'    {LBL[a] if a != "rcmd_v2" else "Full"} & ' + ' & '.join(cells) + ' \\\\')
    lines.append('    \\midrule')
    lines.append('    HD & ' + ' & '.join(INSTANCES) + ' \\\\')
    lines.append('    \\midrule')
    for a in ABLS5:
        cells = []
        for i in INSTANCES:
            v = D[a][i][:, 1]
            cells.append(f'{v.mean():.2f}' + '$\\pm$' + f'{v.std():.2f}')
        lines.append(f'    {LBL[a] if a != "rcmd_v2" else "Full"} & ' + ' & '.join(cells) + ' \\\\')
    lines += ['    \\bottomrule', '  \\end{tabular}}', '\\end{table*}', '']
    (SUPP / 'tab_ablation_detail.tex').write_text('\n'.join(lines), encoding='utf-8',
                                                   newline='\n')
    print('supp tab_ablation_detail done')


# ══════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════
FIGURES = [
    ('fig4', fig4_ranking),
    ('fig5', fig5_convergence),        # 正文已删除；默认跳过（--all 可强制）
    ('fig6', fig6_dual_signal),
    ('fig7', fig7_assignment),
    ('fig8', fig8_ablation),
    ('fig3', fig3_concept),
    ('supp_fig1', supp_fig1_main_comparison),
    ('supp_fig2', supp_fig2_rmi_heatmap),
    ('supp_fig6', supp_fig6_rmi_boxplot),
    ('supp_figS', supp_figS_ablation_detail),
    ('supp_tab', supp_tab_ablation_detail),
]
DEFAULT_SKIP = {'fig5'}   # 正文已删除 fig5，保留现有文件不覆盖


def main():
    ap = argparse.ArgumentParser(description='regenerate revision figures')
    ap.add_argument('--only', nargs='*', default=None,
                    help='only generate these (e.g. fig3 fig4 supp_fig1)')
    ap.add_argument('--skip', nargs='*', default=[], help='skip these')
    ap.add_argument('--all', action='store_true', help='include fig5 (skipped by default)')
    args = ap.parse_args()
    skip = set(args.skip) if args.skip is not None else set()
    if not args.all:
        skip |= DEFAULT_SKIP
    done = []
    for name, fn in FIGURES:
        if args.only is not None and name not in args.only:
            continue
        if name in skip:
            print(f'skip {name}')
            continue
        fn()
        done.append(name)
    print('ALL DONE:', ', '.join(done) if done else '(nothing)')


if __name__ == '__main__':
    main()
