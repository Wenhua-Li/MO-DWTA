"""scripts/analyze_baselines.py — 6个对比基线算法全量实验报告

读取 reports/{nsga2,moead,spea2,commea,hdmmode,mmoeadc}/ 下的JSON结果，
生成含统计检验、箱线图、Pareto前沿可视化的自包含HTML报告。

用法：
  python scripts/analyze_baselines.py [--reports-dir reports] [--out-dir reports/analysis_baselines]
"""
import sys, argparse, json, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64, io

warnings.filterwarnings('ignore')
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── 常量 ─────────────────────────────────────────────────────────────

ALGOS = ['nsga2', 'moead', 'spea2', 'commea', 'hdmmode', 'mmoeadc']
ALGO_LABELS = {
    'nsga2':   'NSGA-II',
    'moead':   'MOEA/D',
    'spea2':   'SPEA2',
    'commea':  'CoMMEA',
    'hdmmode': 'HDMMODE',
    'mmoeadc': 'MMOEADC',
}
ALGO_COLORS = {
    'nsga2':   '#1f77b4',
    'moead':   '#ff7f0e',
    'spea2':   '#2ca02c',
    'commea':  '#d62728',
    'hdmmode': '#9467bd',
    'mmoeadc': '#8c564b',
}
INSTANCES = [f'I{i}' for i in range(1, 11)]
INST_META = {
    'I1':  dict(n_w=6,  n_r=6,  n_t=13,  n_ph=2),
    'I2':  dict(n_w=7,  n_r=6,  n_t=18,  n_ph=3),
    'I3':  dict(n_w=8,  n_r=7,  n_t=24,  n_ph=4),
    'I4':  dict(n_w=8,  n_r=7,  n_t=30,  n_ph=4),
    'I5':  dict(n_w=9,  n_r=7,  n_t=40,  n_ph=5),
    'I6':  dict(n_w=9,  n_r=8,  n_t=60,  n_ph=6),
    'I7':  dict(n_w=10, n_r=8,  n_t=80,  n_ph=7),
    'I8':  dict(n_w=10, n_r=8,  n_t=120, n_ph=8),
    'I9':  dict(n_w=10, n_r=8,  n_t=300, n_ph=9),
    'I10': dict(n_w=10, n_r=8,  n_t=500, n_ph=10),
}

# ── 数据加载 ──────────────────────────────────────────────────────────

def load_all(reports_dir: Path):
    data = defaultdict(lambda: defaultdict(list))
    missing = []
    for algo in ALGOS:
        for inst in INSTANCES:
            for run in range(4):
                p = reports_dir / algo / f'{inst}_run{run:03d}.json'
                if p.exists():
                    with open(p) as f:
                        data[algo][inst].append(json.load(f))
                else:
                    missing.append(str(p))
    if missing:
        print(f'WARNING: {len(missing)} files missing (may still be running)')
    return data

# ── HV ───────────────────────────────────────────────────────────────

def hv2d(points, ref):
    if len(points) == 0:
        return 0.0
    pts = ref - np.array(points)
    pts = pts[(pts[:, 0] > 0) & (pts[:, 1] > 0)]
    if len(pts) == 0:
        return 0.0
    pts = pts[pts[:, 0].argsort()[::-1]]
    hv = 0.0; max_y = 0.0
    for x, y in pts:
        if y > max_y:
            hv += x * (y - max_y); max_y = y
    return float(hv)

def global_ref(data):
    max_f1 = defaultdict(float); max_f2 = defaultdict(float)
    for algo in ALGOS:
        for inst in INSTANCES:
            for rd in data[algo][inst]:
                for ph in rd.get('phases', []):
                    objs = ph.get('pareto_objs', [])
                    if objs:
                        arr = np.array(objs)
                        max_f1[inst] = max(max_f1[inst], arr[:, 0].max())
                        max_f2[inst] = max(max_f2[inst], arr[:, 1].max())
    return {inst: np.array([max_f1[inst] * 1.1 + 1e-6, max_f2[inst] * 1.1 + 1e-6])
            for inst in INSTANCES}

# ── 指标提取 ──────────────────────────────────────────────────────────

def extract_metrics(data, ref_pts):
    """返回 metrics[algo][inst][metric] = np.array(shape=n_runs)"""
    m = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for algo in ALGOS:
        for inst in INSTANCES:
            for rd in data[algo][inst]:
                phases = rd.get('phases', [])
                if not phases:
                    continue
                m[algo][inst]['total_f1'].append(
                    rd.get('total_f1_res', sum(p['best_f1_res'] for p in phases)))
                m[algo][inst]['elapsed_s'].append(rd.get('elapsed_s', np.nan))
                rmis = [p['best_rmi'] for p in phases if 'best_rmi' in p]
                m[algo][inst]['mean_rmi'].append(np.mean(rmis) if rmis else np.nan)
                hds = [p.get('hd', np.nan) for p in phases]
                m[algo][inst]['mean_hd'].append(np.nanmean(hds))
                pns = [p.get('pareto_size', len(p.get('pareto_objs', []))) for p in phases]
                m[algo][inst]['mean_pareto_n'].append(np.mean(pns))
                hv_total = sum(hv2d(np.array(ph['pareto_objs']), ref_pts[inst])
                               for ph in phases if ph.get('pareto_objs'))
                m[algo][inst]['hv'].append(hv_total)
    out = {}
    for algo in ALGOS:
        out[algo] = {}
        for inst in INSTANCES:
            out[algo][inst] = {k: np.array(v, dtype=float)
                               for k, v in m[algo][inst].items()}
    return out

# ── 统计检验 ──────────────────────────────────────────────────────────

def friedman_rank(metrics, metric, min_better=True):
    matrix = []
    for inst in INSTANCES:
        row = []
        ok = True
        for algo in ALGOS:
            v = metrics[algo][inst].get(metric, np.array([]))
            v = v[~np.isnan(v)]
            if len(v) == 0:
                ok = False; break
            row.append(np.median(v))
        if ok:
            matrix.append(row)
    if not matrix:
        return {a: np.nan for a in ALGOS}, np.nan
    mat = np.array(matrix)
    ranks = np.array([stats.rankdata(r if min_better else -r) for r in mat])
    avg_ranks = ranks.mean(axis=0)
    try:
        _, p = stats.friedmanchisquare(*[ranks[:, j] for j in range(len(ALGOS))])
    except Exception:
        p = np.nan
    return dict(zip(ALGOS, avg_ranks)), p

def wilcoxon_vs_nsga2(metrics, metric, min_better=True):
    """每个实例：各算法 vs NSGA-II 的Mann-Whitney U检验符号"""
    result = {}
    ref_algo = 'nsga2'
    for inst in INSTANCES:
        xi = metrics[ref_algo][inst].get(metric, np.array([]))
        xi = xi[~np.isnan(xi)]
        row = {}
        for algo in ALGOS:
            if algo == ref_algo:
                row[algo] = '—'; continue
            xj = metrics[algo][inst].get(metric, np.array([]))
            xj = xj[~np.isnan(xj)]
            if len(xi) < 2 or len(xj) < 2:
                row[algo] = 'N/A'; continue
            _, p = stats.mannwhitneyu(xi, xj, alternative='two-sided')
            if p >= 0.05:
                row[algo] = '='
            elif (np.median(xj) < np.median(xi)) == min_better:
                row[algo] = '+'   # algo 优于 NSGA-II
            else:
                row[algo] = '-'
        result[inst] = row
    return result

# ── 图表 ─────────────────────────────────────────────────────────────

def _b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def plot_boxplots(metrics, metric, ylabel, title, min_better=True):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=False)
    fig.suptitle(title, fontsize=13, fontweight='bold')
    for idx, inst in enumerate(INSTANCES):
        ax = axes[idx // 5][idx % 5]
        data_list, colors = [], []
        for algo in ALGOS:
            v = metrics[algo][inst].get(metric, np.array([]))
            v = v[~np.isnan(v)]
            if len(v):
                data_list.append(v); colors.append(ALGO_COLORS[algo])
        if data_list:
            bp = ax.boxplot(data_list, patch_artist=True,
                            medianprops=dict(color='black', linewidth=2))
            for patch, c in zip(bp['boxes'], colors):
                patch.set_facecolor(c); patch.set_alpha(0.7)
        ax.set_xticklabels([ALGO_LABELS[a][:7] for a in ALGOS
                             if len(metrics[a][inst].get(metric, [])) > 0],
                            fontsize=6, rotation=40)
        ax.set_title(f'{inst} (n_t={INST_META[inst]["n_t"]})', fontsize=9)
        ax.set_ylabel(ylabel if idx % 5 == 0 else '', fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _b64(fig)

def plot_scalability(metrics):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    specs = [
        ('total_f1',  'Total F1 residual (↓)', True),
        ('hv',        'Hypervolume (↑)',        False),
        ('mean_hd',   'Hamming diversity (↑)',  False),
    ]
    n_targets = [INST_META[i]['n_t'] for i in INSTANCES]
    for ax, (mk, ylabel, _) in zip(axes, specs):
        for algo in ALGOS:
            meds = [np.median(v) if len(v := metrics[algo][inst].get(mk, np.array([]))[
                ~np.isnan(metrics[algo][inst].get(mk, np.array([])))]) else np.nan
                    for inst in INSTANCES]
            ax.plot(n_targets, meds, 'o-', color=ALGO_COLORS[algo],
                    label=ALGO_LABELS[algo], lw=2, ms=6)
        ax.set_xlabel('n_t (log scale)', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xscale('log')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _b64(fig)

def plot_pareto_overlay(data, inst='I3'):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'Pareto Fronts — {inst}, best run (min total_f1)', fontsize=12, fontweight='bold')
    for idx, algo in enumerate(ALGOS):
        ax = axes[idx // 3][idx % 3]
        runs = data[algo][inst]
        if not runs:
            ax.set_title(f'{ALGO_LABELS[algo]} (no data)'); continue
        best = min(runs, key=lambda r: r.get('total_f1_res', 1e9))
        for ph in best.get('phases', []):
            objs = ph.get('pareto_objs', [])
            if not objs: continue
            arr = np.array(objs)
            ax.scatter(arr[:, 0], arr[:, 1], s=25, alpha=0.7,
                       label=f'Ph{ph["t"]+1}')
        ax.set_title(ALGO_LABELS[algo], fontsize=11, color=ALGO_COLORS[algo], fontweight='bold')
        ax.set_xlabel('F1 residual threat', fontsize=9)
        ax.set_ylabel('F2 radar load', fontsize=9)
        ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _b64(fig)

def plot_rank_bar(avg_ranks):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(ALGOS))
    bars = ax.bar(x, [avg_ranks.get(a, np.nan) for a in ALGOS],
                  color=[ALGO_COLORS[a] for a in ALGOS], alpha=0.8, edgecolor='black')
    for bar, algo in zip(bars, ALGOS):
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                    f'{h:.2f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_LABELS[a] for a in ALGOS], fontsize=10)
    ax.set_ylabel('Average Friedman rank (lower = better)', fontsize=10)
    ax.set_title('Friedman Average Ranks — Total F1 Residual', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return _b64(fig)


# ── HTML模板 ──────────────────────────────────────────────────────────

CSS = """
body{font-family:Arial,sans-serif;margin:40px;color:#222;background:#fafafa;}
h1{color:#1a1a6e;border-bottom:3px solid #1a1a6e;padding-bottom:8px;}
h2{color:#2c3e7a;margin-top:40px;border-left:5px solid #2c3e7a;padding-left:10px;}
h3{color:#444;margin-top:20px;}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:12px;}
th{background:#2c3e7a;color:white;padding:6px 8px;text-align:center;}
td{padding:4px 8px;border:1px solid #ccc;text-align:center;}
tr:nth-child(even){background:#f0f4ff;}
.best{font-weight:bold;color:#c00;}
.win{color:#1a8a1a;font-weight:bold;}
.lose{color:#c00;}
.tie{color:#888;}
img{max-width:100%;margin:10px 0;border:1px solid #ddd;border-radius:4px;}
.note{background:#e8f0fe;border-left:4px solid #2c3e7a;padding:10px 16px;
      margin:12px 0;border-radius:4px;font-size:13px;}
"""


def _tbl_header(cols):
    return '<tr>' + ''.join(f'<th>{c}</th>' for c in cols) + '</tr>'


def build_summary_html(metrics, metric, min_better=True):
    best = {}
    for inst in INSTANCES:
        vals = {a: np.nanmedian(metrics[a][inst].get(metric, np.array([]))) for a in ALGOS}
        valid = {a: v for a, v in vals.items() if not np.isnan(v)}
        if valid:
            best[inst] = min(valid, key=valid.get) if min_better else max(valid, key=valid.get)
    rows = []
    for algo in ALGOS:
        cells = [f'<td><b>{ALGO_LABELS[algo]}</b></td>']
        for inst in INSTANCES:
            v = metrics[algo][inst].get(metric, np.array([]))
            v = v[~np.isnan(v)]
            if len(v) == 0:
                cells.append('<td>N/A</td>')
            else:
                s = f'{np.mean(v):.3f}&plusmn;{np.std(v):.3f}'
                cls = ' class="best"' if best.get(inst) == algo else ''
                cells.append(f'<td{cls}>{s}</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    return f'<table>{_tbl_header(["Algorithm"] + INSTANCES)}{"".join(rows)}</table>'


def build_wilcoxon_html(wx):
    rows = []
    for algo in ALGOS:
        cells = [f'<td><b>{ALGO_LABELS[algo]}</b></td>']
        for inst in INSTANCES:
            s = wx.get(inst, {}).get(algo, 'N/A')
            if s == '+':   cells.append('<td class="win">+</td>')
            elif s == '-': cells.append('<td class="lose">-</td>')
            elif s == '=': cells.append('<td class="tie">=</td>')
            else:          cells.append(f'<td>{s}</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    return f'<table>{_tbl_header(["Algorithm"] + INSTANCES)}{"".join(rows)}</table>'


def build_rank_html(avg_ranks):
    sorted_algos = sorted(ALGOS, key=lambda a: avg_ranks.get(a, 99))
    rows = []
    for i, a in enumerate(sorted_algos):
        cls = ' class="best"' if i == 0 else ''
        v = avg_ranks.get(a, float('nan'))
        rows.append(f'<tr><td><b>{ALGO_LABELS[a]}</b></td><td{cls}>{v:.3f}</td></tr>')
    return f'<table>{_tbl_header(["Algorithm", "Avg. Rank"])}{"".join(rows)}</table>'


def generate_report(reports_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print('Loading data...', flush=True)
    data = load_all(reports_dir)
    ref_pts = global_ref(data)
    print('Extracting metrics...', flush=True)
    metrics = extract_metrics(data, ref_pts)

    ranks_f1, p_f1 = friedman_rank(metrics, 'total_f1', min_better=True)
    ranks_hv,  p_hv  = friedman_rank(metrics, 'hv',      min_better=False)
    wx_f1  = wilcoxon_vs_nsga2(metrics, 'total_f1', min_better=True)
    wx_hv  = wilcoxon_vs_nsga2(metrics, 'hv',       min_better=False)
    wx_hd  = wilcoxon_vs_nsga2(metrics, 'mean_hd',  min_better=False)
    wx_rmi = wilcoxon_vs_nsga2(metrics, 'mean_rmi', min_better=False)

    print('Generating figures...', flush=True)
    fig_f1   = plot_boxplots(metrics, 'total_f1',  'Total F1 residual (lower=better)', 'Total F1 Residual')
    fig_hv   = plot_boxplots(metrics, 'hv',        'Hypervolume (higher=better)',      'Hypervolume',   min_better=False)
    fig_hd   = plot_boxplots(metrics, 'mean_hd',   'Hamming diversity (higher=better)','Hamming Diversity', min_better=False)
    fig_rmi  = plot_boxplots(metrics, 'mean_rmi',  'RMI (higher=better)',              'Post-Selection RMI', min_better=False)
    fig_scal = plot_scalability(metrics)
    fig_pf   = plot_pareto_overlay(data, inst='I3')
    fig_rank = plot_rank_bar(ranks_f1)

    rows = []
    for algo in ALGOS:
        for inst in INSTANCES:
            for mk in ['total_f1', 'hv', 'mean_hd', 'mean_rmi', 'mean_pareto_n', 'elapsed_s']:
                v = metrics[algo][inst].get(mk, np.array([]))
                v = v[~np.isnan(v)]
                if len(v):
                    rows.append({'algo': algo, 'instance': inst, 'metric': mk,
                                 'mean': np.mean(v), 'std': np.std(v),
                                 'median': np.median(v), 'n_runs': len(v)})
    pd.DataFrame(rows).to_csv(out_dir / 'baseline_stats.csv', index=False)

    n_runs_actual = sum(len(data[a][i]) for a in ALGOS for i in INSTANCES)
    inst_rows = ''.join(
        f'<tr><td>{i}</td><td>{m["n_w"]}</td><td>{m["n_r"]}</td>'
        f'<td>{m["n_t"]}</td><td>{m["n_ph"]}</td></tr>'
        for i, m in INST_META.items()
    )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Baseline Comparison Report</title>
<style>{CSS}</style></head><body>
<h1>Multi-Modal Multi-Objective WTA &mdash; Baseline Algorithm Comparison</h1>
<div class="note">
<b>Setup:</b> {len(ALGOS)} algorithms &times; 10 instances &times; 4 independent runs each.
G_max=100, N=5&times;n_t. Post-selection: TOPSIS (equal weights). Total result files loaded: {n_runs_actual}.<br>
<b>Symbols:</b> <span class="win">+</span> = significantly better than NSGA-II (p&lt;0.05, Mann-Whitney U, two-sided);
<span class="lose">&minus;</span> = significantly worse; <span class="tie">=</span> = no significant difference.
Best median per instance highlighted in <span class="best">red bold</span>.
</div>

<h2>1. Problem Instances</h2>
<table><tr><th>Instance</th><th>Weapons</th><th>Radars</th><th>Targets</th><th>Phases</th></tr>
{inst_rows}</table>

<h2>2. Total F1 Residual Threat (lower = better)</h2>
<img src="data:image/png;base64,{fig_f1}">
{build_summary_html(metrics, 'total_f1', min_better=True)}
<h3>Wilcoxon vs NSGA-II</h3>
{build_wilcoxon_html(wx_f1)}

<h2>3. Hypervolume (higher = better)</h2>
<img src="data:image/png;base64,{fig_hv}">
{build_summary_html(metrics, 'hv', min_better=False)}
<h3>Wilcoxon vs NSGA-II</h3>
{build_wilcoxon_html(wx_hv)}

<h2>4. Hamming Diversity HD (higher = better)</h2>
<img src="data:image/png;base64,{fig_hd}">
{build_summary_html(metrics, 'mean_hd', min_better=False)}
<h3>Wilcoxon vs NSGA-II</h3>
{build_wilcoxon_html(wx_hd)}

<h2>5. Post-Selection RMI (higher = better)</h2>
<img src="data:image/png;base64,{fig_rmi}">
{build_summary_html(metrics, 'mean_rmi', min_better=False)}
<h3>Wilcoxon vs NSGA-II</h3>
{build_wilcoxon_html(wx_rmi)}

<h2>6. Scalability Analysis</h2>
<img src="data:image/png;base64,{fig_scal}">

<h2>7. Pareto Front Visualization (Instance I3, best run per algorithm)</h2>
<img src="data:image/png;base64,{fig_pf}">

<h2>8. Friedman Rankings</h2>
<h3>Total F1 Residual (Friedman p={p_f1:.4f})</h3>
<img src="data:image/png;base64,{fig_rank}">
{build_rank_html(ranks_f1)}
<h3>Hypervolume (Friedman p={p_hv:.4f})</h3>
{build_rank_html(ranks_hv)}

</body></html>"""

    out_path = out_dir / 'baseline_report.html'
    out_path.write_text(html, encoding='utf-8')
    print(f'Report saved: {out_path}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--reports-dir', default='reports')
    p.add_argument('--out-dir',     default='reports/analysis_baselines')
    args = p.parse_args()
    rdir = Path(args.reports_dir)
    if not rdir.is_absolute(): rdir = ROOT / rdir
    odir = Path(args.out_dir)
    if not odir.is_absolute(): odir = ROOT / odir
    generate_report(rdir, odir)
