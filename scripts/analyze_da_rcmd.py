"""analyze_da_rcmd.py — RCMD-EA vs 基线的全量实验分析

读取 reports/ 下的 JSON 结果，计算：
- 各指标（HV/HD/RMI/F1）的均值±std
- RCMD-EA vs NSGA-II+TOPSIS 的相对改善
- Wilcoxon 显著性检验
- Friedman 排名

输出到 reports/da_rcmd_full_YYYYMMDD/
"""
import sys, json, re
from pathlib import Path
from datetime import datetime
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
RESULTS_DIR = ROOT / 'reports'

ALGOS = ['da_rcmd', 'nsga2_topsis', 'nsga2_vikor', 'kgdrea']
ALGO_LABELS = {
    'da_rcmd':      'RCMD-EA',
    'nsga2_topsis': 'NSGA-II+TOPSIS',
    'nsga2_vikor':  'NSGA-II+VIKOR',
    'kgdrea':       'KG-DREA',
}
INSTANCES = [f'I{i}' for i in range(1, 11)]


def load_results(algo, inst, n_runs=30):
    """加载指定算法/实例的所有运行结果"""
    algo_dir = RESULTS_DIR / algo
    results = []
    for run in range(n_runs):
        path = algo_dir / f'{inst}_run{run:03d}.json'
        if path.exists():
            with open(path) as f:
                results.append(json.load(f))
    return results


def extract_metrics(results, algo):
    """从结果列表中提取关键指标"""
    total_f1 = []
    mean_rmi = []
    mean_hd = []
    mean_pareto_size = []
    mean_hv = []

    for r in results:
        total_f1.append(r.get('total_f1_res', float('nan')))
        phases = r.get('phases', [])
        if not phases:
            continue
        rmi_vals = [p.get('best_rmi', float('nan')) for p in phases if 'best_rmi' in p]
        hd_vals  = [p.get('hd', 0.0) for p in phases]
        ps_vals  = [p.get('pareto_size', 0) for p in phases]
        hv_vals  = [p.get('hv', 0.0) for p in phases if 'hv' in p]

        mean_rmi.append(np.nanmean(rmi_vals) if rmi_vals else float('nan'))
        mean_hd.append(np.nanmean(hd_vals) if hd_vals else 0.0)
        mean_pareto_size.append(np.nanmean(ps_vals) if ps_vals else 0.0)
        mean_hv.append(np.nanmean(hv_vals) if hv_vals else 0.0)

    return {
        'total_f1_res': np.array(total_f1),
        'mean_rmi': np.array(mean_rmi),
        'mean_hd': np.array(mean_hd),
        'mean_pareto_size': np.array(mean_pareto_size),
        'mean_hv': np.array(mean_hv),
    }


def wilcoxon_test(a, b):
    """Wilcoxon 符号秩检验，返回 (statistic, p_value, symbol)"""
    from scipy.stats import wilcoxon
    a, b = np.array(a), np.array(b)
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]
    if len(a) < 5:
        return float('nan'), float('nan'), '?'
    diff = a - b
    if np.all(diff == 0):
        return 0.0, 1.0, '='
    try:
        stat, p = wilcoxon(diff)
        if p < 0.05:
            symbol = '+' if np.median(diff) > 0 else '-'
        else:
            symbol = '='
        return float(stat), float(p), symbol
    except Exception:
        return float('nan'), float('nan'), '?'


def friedman_rank(data_dict, metric, higher_better=True):
    """Friedman 排名，data_dict: {algo: {inst: array}}"""
    algos = list(data_dict.keys())
    instances = list(next(iter(data_dict.values())).keys())
    ranks = {a: [] for a in algos}
    for inst in instances:
        vals = []
        for a in algos:
            m = data_dict[a].get(inst, {}).get(metric, np.array([]))
            v = np.nanmedian(m) if len(m) > 0 else float('nan')
            vals.append(v)
        # 排名（1 = 最好）
        order = np.argsort(vals if not higher_better else [-v for v in vals])
        for rank, idx in enumerate(order):
            ranks[algos[idx]].append(rank + 1)
    return {a: np.mean(ranks[a]) for a in algos}


def main():
    batch = datetime.now().strftime('%Y-%m-%d')
    out_dir = RESULTS_DIR / f'da_rcmd_full_{batch}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载所有数据
    all_data = {}
    for algo in ALGOS:
        all_data[algo] = {}
        for inst in INSTANCES:
            results = load_results(algo, inst, n_runs=30)
            if results:
                all_data[algo][inst] = extract_metrics(results, algo)
                n = len(results)
                if n < 30:
                    print(f'  警告: {algo}/{inst} 只有 {n}/30 条结果')

    # 2. 统计汇总表
    metrics_info = [
        ('total_f1_res', '总残余威胁', False),   # 越小越好
        ('mean_rmi',     '平均RMI',   True),    # 越大越好
        ('mean_hd',      '平均HD',    True),    # 越大越好
        ('mean_pareto_size', 'Pareto集大小', True),
        ('mean_hv',      '平均HV',    True),    # 越大越好（仅da_rcmd有）
    ]

    print('\n' + '='*90)
    print('RCMD-EA 全量实验结果（各实例均值±std，30次）')
    print('='*90)

    summary_lines = []
    for metric, label, higher_better in metrics_info:
        print(f'\n--- {label} ({"↑大好" if higher_better else "↓小好"}) ---')
        header = f'{"实例":<6}' + ''.join(f'{ALGO_LABELS[a]:<20}' for a in ALGOS if metric in all_data[a].get('I1', {}))
        print(header)

        for inst in INSTANCES:
            row = f'{inst:<6}'
            vals_per_algo = {}
            for algo in ALGOS:
                m = all_data[algo].get(inst, {}).get(metric, np.array([]))
                valid = m[~np.isnan(m)]
                if len(valid) > 0:
                    row += f'{np.mean(valid):8.3f}±{np.std(valid):6.3f}     '
                    vals_per_algo[algo] = valid
                else:
                    row += f'{"N/A":<20}'
            print(row)

    # 3. RCMD-EA vs TOPSIS 显著性检验（每实例每指标）
    print('\n\n' + '='*90)
    print('RCMD-EA vs NSGA-II+TOPSIS 显著性检验（Wilcoxon，+好/-差/=无差）')
    print('='*90)

    sig_results = {}
    for metric, label, higher_better in metrics_info:
        print(f'\n{label}:')
        sig_row = {}
        for inst in INSTANCES:
            m_da = all_data['da_rcmd'].get(inst, {}).get(metric, np.array([]))
            m_tp = all_data['nsga2_topsis'].get(inst, {}).get(metric, np.array([]))
            if len(m_da) < 5 or len(m_tp) < 5:
                sig_row[inst] = '?'
                continue
            if not higher_better:
                stat, p, sym = wilcoxon_test(-m_da, -m_tp)
            else:
                stat, p, sym = wilcoxon_test(m_da, m_tp)
            sig_row[inst] = sym
            da_med = np.nanmedian(m_da)
            tp_med = np.nanmedian(m_tp)
            diff = (da_med - tp_med) / (abs(tp_med) + 1e-12) * 100
            print(f'  {inst}: {sym} (p={p:.4f}, DA_med={da_med:.3f}, TOPSIS_med={tp_med:.3f}, diff={diff:+.1f}%)')
        sig_results[metric] = sig_row

    # 4. Friedman 排名
    print('\n\n' + '='*90)
    print('Friedman 排名（1=最优）')
    print('='*90)
    for metric, label, higher_better in metrics_info:
        ranks = friedman_rank(all_data, metric, higher_better)
        rank_str = '  '.join(f'{ALGO_LABELS[a]}: {ranks[a]:.2f}' for a in sorted(ranks, key=lambda x: ranks[x]))
        print(f'{label}: {rank_str}')

    # 5. 与旧版 RCMD-EA 的关键对比（DA vs 原版，HV 改善核心指标）
    print('\n\n' + '='*90)
    print('HV 改善汇总（RCMD-EA vs 原版 RCMD-EA）')
    print('='*90)
    if 'rcmd_ea' in all_data and 'da_rcmd' in all_data:
        for inst in INSTANCES:
            m_da = all_data['da_rcmd'].get(inst, {}).get('mean_hv', np.array([]))
            m_orig = all_data.get('rcmd_ea', {}).get(inst, {}).get('mean_pareto_size', np.array([]))
            if len(m_da) > 0:
                # 用 pareto_size 比较（原版无 hv 字段）
                ps_da = np.nanmean(all_data['da_rcmd'].get(inst, {}).get('mean_pareto_size', np.array([])))
                ps_nsga = np.nanmean(all_data['nsga2_topsis'].get(inst, {}).get('mean_pareto_size', np.array([])))
                hv_da = np.nanmean(m_da)
                print(f'  {inst}: DA Pareto_size={ps_da:.1f}, NSGA Pareto_size={ps_nsga:.1f}, DA HV_mean={hv_da:.3f}')

    print(f'\n分析完成，结果目录: {out_dir}')

    # 保存 CSV
    import csv
    with open(out_dir / 'da_rcmd_summary.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['算法', '实例', 'total_f1_mean', 'total_f1_std',
                         'mean_rmi_mean', 'mean_rmi_std', 'mean_hd_mean', 'mean_hd_std',
                         'pareto_size_mean', 'hv_mean'])
        for algo in ALGOS:
            for inst in INSTANCES:
                m = all_data[algo].get(inst, {})
                def ms(key):
                    v = m.get(key, np.array([]))
                    valid = v[~np.isnan(v)]
                    return (f'{np.mean(valid):.4f}', f'{np.std(valid):.4f}') if len(valid) > 0 else ('', '')
                f1m, f1s = ms('total_f1_res')
                rm, rs = ms('mean_rmi')
                hm, hs = ms('mean_hd')
                pm, _ = ms('mean_pareto_size')
                hvm, _ = ms('mean_hv')
                writer.writerow([ALGO_LABELS[algo], inst, f1m, f1s, rm, rs, hm, hs, pm, hvm])

    print(f'CSV 已保存: {out_dir / "da_rcmd_summary.csv"}')


if __name__ == '__main__':
    main()
