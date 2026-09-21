"""summarize_rmi_fix.py — RMI公式修正后的快速核对报告

读取 reports/<algo>/I*_run*.json，对每个算法计算跨10实例×30次运行的
RMI均值/F1均值，与Friedman平均排名，用于确认新固定参考点公式下
RCMD-EA相对基线的优势方向是否依然成立。

不做正式论文表格（那部分仍需analyze_baselines.py等按新目录名调整后生成），
仅做本次修复的快速数值核验。
"""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats

ROOT = Path(__file__).parent.parent
REPORTS = ROOT / 'reports'

ALGOS = ['rcmd_ea', 'nsga2_topsis', 'nsga2_vikor', 'moead_topsis', 'spea2_topsis',
          'commea_topsis', 'hdmmode_topsis', 'mmoeadc_topsis',
          'abl_no_explore', 'abl_fixed_pref', 'abl_no_dual_archive']
INSTANCES = [f'I{i}' for i in range(1, 11)]


def load_metric(algo, inst):
    """返回该算法在该实例上30次run的(mean_rmi, mean_f1)列表"""
    rmis, f1s = [], []
    for run in range(30):
        p = REPORTS / algo / f'{inst}_run{run:03d}.json'
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding='utf-8'))
        phases = d.get('phases', [])
        if not phases:
            continue
        rmi_vals = [ph['best_rmi'] for ph in phases if 'best_rmi' in ph]
        if rmi_vals:
            rmis.append(float(np.mean(rmi_vals)))
        f1s.append(float(d.get('total_f1_res', np.nan)))
    return rmis, f1s


def main():
    print(f"{'Algo':20s} {'meanRMI(all inst)':>18s} {'meanF1(all inst)':>16s}")
    print('-' * 58)
    rmi_table = {}
    f1_table = {}
    for algo in ALGOS:
        all_rmi, all_f1 = [], []
        per_inst_rmi = {}
        for inst in INSTANCES:
            rmis, f1s = load_metric(algo, inst)
            if rmis:
                per_inst_rmi[inst] = float(np.mean(rmis))
                all_rmi.extend(rmis)
            if f1s:
                all_f1.extend(f1s)
        rmi_table[algo] = per_inst_rmi
        f1_table[algo] = all_f1
        mrmi = np.mean(all_rmi) if all_rmi else float('nan')
        mf1 = np.mean(all_f1) if all_f1 else float('nan')
        print(f"{algo:20s} {mrmi:18.3f} {mf1:16.3f}")

    print()
    print("=== RCMD-EA vs 每个基线：逐实例RMI均值对比（新公式）===")
    baselines = [a for a in ALGOS if a not in ('rcmd_ea', 'abl_no_explore', 'abl_fixed_pref', 'abl_no_dual_archive')]
    for inst in INSTANCES:
        rcmd_v = rmi_table['rcmd_ea'].get(inst, float('nan'))
        row = f"{inst:5s} RCMD-EA={rcmd_v:9.3f} | "
        wins = 0
        for b in baselines:
            bv = rmi_table[b].get(inst, float('nan'))
            win = rcmd_v > bv
            wins += int(win)
            row += f"{b}={bv:8.3f}{'*' if win else ' '} "
        print(row + f"  (RCMD-EA领先{wins}/{len(baselines)}个基线)")

    print()
    print("=== Friedman检验（跨10实例，7个主要算法的RMI均值排名）===")
    main_algos = ['rcmd_ea', 'nsga2_topsis', 'nsga2_vikor', 'moead_topsis',
                  'spea2_topsis', 'commea_topsis', 'hdmmode_topsis', 'mmoeadc_topsis']
    data_matrix = []
    valid_insts = [inst for inst in INSTANCES if all(inst in rmi_table[a] for a in main_algos)]
    for inst in valid_insts:
        data_matrix.append([rmi_table[a][inst] for a in main_algos])
    data_matrix = np.array(data_matrix)  # (n_inst, n_algo)
    if len(valid_insts) >= 3:
        stat, p = stats.friedmanchisquare(*[data_matrix[:, i] for i in range(len(main_algos))])
        # 平均排名（RMI越大越好，故按降序排名，rank1=最好）
        ranks = np.array([stats.rankdata(-row) for row in data_matrix])  # 降序rank
        mean_ranks = ranks.mean(axis=0)
        print(f"Friedman chi2={stat:.3f}, p={p:.6f}  (n_instances={len(valid_insts)})")
        for a, r in sorted(zip(main_algos, mean_ranks), key=lambda x: x[1]):
            print(f"  {a:20s} mean_rank={r:.2f}")
    else:
        print("有效实例数不足，跳过Friedman检验")


if __name__ == '__main__':
    main()
