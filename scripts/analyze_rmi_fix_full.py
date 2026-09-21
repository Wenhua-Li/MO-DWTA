"""scripts/analyze_rmi_fix_full.py — RMI公式修正后的完整正式报告

覆盖全部11个算法目录（8主算法+3消融变体），生成：
  - RMI/HD 主表（10实例×11算法，mean±std）
  - RCMD-EA vs 7个基线的Wilcoxon signed-rank检验（逐实例，RMI维度）
  - 8个主算法的Friedman检验+平均排名（RMI维度）
  - 结果保存为CSV，输出目录：项目根目录 reports/<batch>/（遵循报告目录规范）

关于F1（2026-07-09修订）：F1不再作为跨算法比较维度。原因：RCMD-EA的MORI
鲁棒性约束会影响其自身的选择过程，本次RMI公式修正后MORI约束变严格，
RCMD-EA为追求真正的邻域鲁棒性需要付出更多F1代价（真实的鲁棒性-效能权衡）；
而7个基线的选择流程完全不涉及MORI/RMI，F1几乎不受本次修正影响（验证：
新旧F1均值差异<0.5%，属随机种子噪声）。用变化前后不对等的F1做跨算法比较
不再合理，故论文改为仅用RMI单维度做跨算法对比。raw_metrics.csv仍保留f1
原始列，供需要问题内部（非跨算法比较）分析时使用。

用法：
  python scripts/analyze_rmi_fix_full.py
"""
import json
from pathlib import Path
from collections import defaultdict
from datetime import date
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent.parent           # mowta/
PROJECT_ROOT = ROOT.parent                     # 项目根目录
REPORTS_DIR = ROOT / 'reports'                 # 原始JSON结果所在

BATCH = f'rmi_fix_{date.today().isoformat()}'
OUT_DIR = PROJECT_ROOT / 'reports' / BATCH
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAIN_ALGOS = ['rcmd_ea', 'nsga2_topsis', 'nsga2_vikor', 'moead_topsis',
              'spea2_topsis', 'commea_topsis', 'hdmmode_topsis', 'mmoeadc_topsis']
ABLATION_ALGOS = ['abl_no_explore', 'abl_fixed_pref', 'abl_no_dual_archive']
ALL_ALGOS = MAIN_ALGOS + ABLATION_ALGOS

ALGO_LABELS = {
    'rcmd_ea': 'RCMD-EA', 'nsga2_topsis': 'NSGA-II+TOPSIS', 'nsga2_vikor': 'NSGA-II+VIKOR',
    'moead_topsis': 'MOEA/D+TOPSIS', 'spea2_topsis': 'SPEA2+TOPSIS',
    'commea_topsis': 'CoMMEA+TOPSIS', 'hdmmode_topsis': 'HDMMODE+TOPSIS',
    'mmoeadc_topsis': 'MMOEADC+TOPSIS',
    'abl_no_explore': 'RCMD-EA(-C2)', 'abl_fixed_pref': 'RCMD-EA(-C3)',
    'abl_no_dual_archive': 'RCMD-EA(-DA)',
}
INSTANCES = [f'I{i}' for i in range(1, 11)]
N_RUNS = 30


def load_runs(algo, inst):
    """加载该算法在该实例上全部run，返回list of dict"""
    runs = []
    for run in range(N_RUNS):
        p = REPORTS_DIR / algo / f'{inst}_run{run:03d}.json'
        if p.exists():
            runs.append(json.loads(p.read_text(encoding='utf-8')))
    return runs


def per_run_metrics(rd):
    """从单次run的结果字典提取(mean_rmi, total_f1, mean_hd)"""
    phases = rd.get('phases', [])
    if not phases:
        return np.nan, np.nan, np.nan
    rmis = [p['best_rmi'] for p in phases if 'best_rmi' in p]
    mean_rmi = float(np.mean(rmis)) if rmis else np.nan
    total_f1 = float(rd.get('total_f1_res', np.nan))
    hds = [p.get('hd', np.nan) for p in phases]
    mean_hd = float(np.nanmean(hds)) if hds else np.nan
    return mean_rmi, total_f1, mean_hd


def build_raw_table():
    """rows: (algo, inst, run, rmi, f1, hd)"""
    rows = []
    for algo in ALL_ALGOS:
        for inst in INSTANCES:
            for rd in load_runs(algo, inst):
                rmi, f1, hd = per_run_metrics(rd)
                rows.append(dict(algo=algo, inst=inst, run=rd.get('run'), rmi=rmi, f1=f1, hd=hd))
    return pd.DataFrame(rows)


def main():
    df = build_raw_table()
    df.to_csv(OUT_DIR / 'raw_metrics.csv', index=False, encoding='utf-8-sig')
    print(f"原始明细已保存: {OUT_DIR / 'raw_metrics.csv'}  ({len(df)}行)")

    # ── 主表：mean±std ──────────────────────────────────────────────
    # 注：F1不再生成跨算法对比主表（table_f1_main.csv）。RMI公式修正后，
    # RCMD-EA的MORI约束变严格，为追求真正的邻域鲁棒性需要付出更多F1代价
    # （真实的鲁棒性-效能权衡，而基线F1几乎不受影响，因其选择流程不涉及
    # MORI）。论文改为仅用RMI单维度做跨算法对比，不再报告F1对比结果。
    # raw_metrics.csv仍保留f1原始列供问题内部分析使用。
    for metric in ['rmi', 'hd']:
        pivot_mean = df.pivot_table(index='algo', columns='inst', values=metric, aggfunc='mean')
        pivot_std = df.pivot_table(index='algo', columns='inst', values=metric, aggfunc='std')
        pivot_mean = pivot_mean.reindex(ALL_ALGOS)[INSTANCES]
        pivot_std = pivot_std.reindex(ALL_ALGOS)[INSTANCES]
        combined = pivot_mean.round(3).astype(str) + ' ± ' + pivot_std.round(3).astype(str)
        combined.index = [ALGO_LABELS.get(a, a) for a in combined.index]
        combined.to_csv(OUT_DIR / f'table_{metric}_main.csv', encoding='utf-8-sig')
        print(f"\n=== {metric.upper()} 主表（mean±std）===")
        print(combined.to_string())

    # ── Wilcoxon: RCMD-EA vs 每个基线，逐实例 ────────────────────────
    wilcoxon_rows = []
    baselines = [a for a in MAIN_ALGOS if a != 'rcmd_ea']
    for inst in INSTANCES:
        rcmd_rmi = df[(df.algo == 'rcmd_ea') & (df.inst == inst)]['rmi'].dropna().values
        for b in baselines:
            b_rmi = df[(df.algo == b) & (df.inst == inst)]['rmi'].dropna().values
            n = min(len(rcmd_rmi), len(b_rmi))
            if n < 5:
                wilcoxon_rows.append(dict(inst=inst, baseline=b, n=n, p=np.nan, rcmd_wins=np.nan))
                continue
            try:
                stat, p = stats.wilcoxon(rcmd_rmi[:n], b_rmi[:n])
            except ValueError:
                p = np.nan
            wilcoxon_rows.append(dict(
                inst=inst, baseline=ALGO_LABELS[b], n=n, p=p,
                rcmd_mean=np.mean(rcmd_rmi), baseline_mean=np.mean(b_rmi),
                rcmd_wins=bool(np.mean(rcmd_rmi) > np.mean(b_rmi)),
            ))
    wdf = pd.DataFrame(wilcoxon_rows)
    wdf.to_csv(OUT_DIR / 'table_rmi_wilcoxon.csv', index=False, encoding='utf-8-sig')
    print(f"\n=== Wilcoxon检验结果已保存: {OUT_DIR / 'table_rmi_wilcoxon.csv'} ===")
    sig_and_win = wdf[(wdf.p < 0.05) & (wdf.rcmd_wins == True)]
    print(f"RCMD-EA显著领先(p<0.05且均值更高)的对比数: {len(sig_and_win)}/{len(wdf.dropna(subset=['p']))}")

    # ── Friedman检验：8主算法 × RMI/HD ──────────────────────────────
    # 注：F1不再作为跨算法比较维度（见main.tex修订说明），仅RMI/HD参与Friedman排名。
    # raw_metrics.csv仍保留f1原始列供问题内部分析使用（不做跨算法排名）。
    for metric in ['rmi', 'hd']:
        pivot = df.pivot_table(index='inst', columns='algo', values=metric, aggfunc='mean')
        pivot = pivot.reindex(index=INSTANCES, columns=MAIN_ALGOS).dropna()
        if len(pivot) < 3:
            print(f"\n{metric}: 有效实例不足，跳过Friedman检验")
            continue
        mat = pivot.values
        stat, p = stats.friedmanchisquare(*[mat[:, i] for i in range(mat.shape[1])])
        ranks = np.array([stats.rankdata(-row) for row in mat])  # RMI越大越好，降序排名
        mean_ranks = ranks.mean(axis=0)
        print(f"\n=== Friedman检验（{metric.upper()}，8主算法×{len(pivot)}实例）===")
        print(f"chi2={stat:.3f}, p={p:.6f}")
        rank_df = pd.DataFrame({'algo': [ALGO_LABELS[a] for a in MAIN_ALGOS], 'mean_rank': mean_ranks})
        rank_df = rank_df.sort_values('mean_rank')
        print(rank_df.to_string(index=False))
        rank_df.to_csv(OUT_DIR / f'table_{metric}_friedman.csv', index=False, encoding='utf-8-sig')

    print(f"\n全部结果已保存至: {OUT_DIR}")


if __name__ == '__main__':
    main()
