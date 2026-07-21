"""demo.py — MO-DWTA 快速开始示例

在 I1 算例上运行一次 RCMD-EA，输出关键指标并保存 Pareto 前沿示例图。
"""
import sys
from pathlib import Path
import time

# 将仓库根目录加入 Python 路径，以便使用绝对导入
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from alg_code.problems.instance import load_instance
from alg_code.rcmd_ea import run_rcmd_ea, RCMDParams
from alg_code.metrics import hypervolume, compute_reference_point


def main():
    inst_path = ROOT / 'instances' / 'I1.mat'
    print(f"[Demo] Loading instance from {inst_path}")
    inst = load_instance(str(inst_path))
    print(f"       Instance: {inst.name}, phases: {inst.T_total}, total targets: {sum(p.n_t for p in inst.phases)}")

    # 轻量参数：G_max=40 用于快速演示；论文实验使用 G_max=100
    params = RCMDParams(
        N_factor=5,
        G_max=40,
        explore_ratio=0.4,
        seed=42,
        G_max_dynamic=None,  # 关闭动态预算，保持 demo 简洁
        warm_start=False,
    )

    print(f"[Demo] Running RCMD-EA (G_max={params.G_max}, seed={params.seed}) ...")
    t0 = time.time()
    result = run_rcmd_ea(inst, params=params, verbose=True)
    elapsed = time.time() - t0

    # 汇总指标
    print("\n========== RCMD-EA Demo Results ==========")
    print(f"Total residual threat (F1): {result.total_f1_res:.4f}")

    hv_list = []
    for pr in result.phase_results:
        objs = pr.pareto_objs
        if objs is None or len(objs) == 0:
            continue
        ref = compute_reference_point(objs)
        hv = hypervolume(objs, ref)
        hv_list.append(hv)
        print(f"Phase {pr.t + 1}: HV={hv:.4f}, "
              f"best F1(res)={pr.best_f1_res:.4f}, "
              f"best F2={pr.best_f2:.4f}, "
              f"RMI={pr.best_rmi:.4f}, "
              f"HD={pr.hd:.4f}")

    print(f"\nMean HV over phases: {np.mean(hv_list):.4f}")
    print(f"Elapsed time: {elapsed:.2f} s")
    print("==========================================\n")

    # 绘制最后阶段的 Pareto 前沿
    last_pr = result.phase_results[-1]
    objs = last_pr.pareto_objs
    if objs is not None and len(objs) > 0:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(objs[:, 0], objs[:, 1], c='tab:red', edgecolors='k', s=60, alpha=0.8, label='RCMD-EA Pareto front')
        ax.scatter(last_pr.best_f1_res, last_pr.best_f2, c='gold', edgecolors='k', s=120, marker='*', zorder=5, label='Selected solution')
        ax.set_xlabel('Residual threat $F_1$ (minimize)')
        ax.set_ylabel('Ammo consumption $F_2$ (minimize)')
        ax.set_title(f'{inst.name} — Phase {last_pr.t + 1} Pareto Front')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.4)
        out_path = ROOT / 'assets' / 'demo_result.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[Demo] Saved sample Pareto front figure to {out_path}")


if __name__ == '__main__':
    main()
