"""baselines/hdmmode.py — HDMMODE 双子种群DE + 局部档案，适配WTA离散问题

Reference: HDMMODE dual-subpopulation DE with archive (see competitor_alg_ref/HDMMODE).
子种群1：收敛导向，自适应规模；子种群2：多样性导向，对子种群1档案区域施加惩罚。
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from alg_code.problems.instance import Instance, PhaseData
from alg_code.problems.encoding import build_combo_set, decode, repair, random_individual
from alg_code.problems.evaluator import evaluate_f1_raw_batch, evaluate_f2_raw_batch
from alg_code.metrics import pareto_dominance_filter, hamming_diversity
from alg_code.baselines._mmea_utils import (
    nd_sort, box_ncm, de_int, hamming_crowd,
)


@dataclass
class HDMMODEParams:
    N_factor: int = 6          # 总种群 = N_factor * n_t（被K=2整除）
    G_max: int = 100
    delta: int = 3             # NCM最小簇大小
    F_de: float = 0.5          # DE缩放因子
    conv_window: int = 20      # 收敛检测窗口
    conv_threshold: float = 0.01
    seed: Optional[int] = None


@dataclass
class HDMMODEPhaseResult:
    t: int
    pareto_objs: np.ndarray
    pareto_codes: np.ndarray
    hd: float


@dataclass
class HDMMODEResult:
    instance_name: str
    phase_results: List[HDMMODEPhaseResult] = field(default_factory=list)


def _build_local_archive(pop, f1, f2, max_arc, delta, combo_sets):
    """NCM聚类后提取各簇局部Pareto代表，并与当前档案合并更新。"""
    clusters = box_ncm(pop, r_frac=0.1)
    local_idx = []
    for cl in clusters:
        if len(cl) <= delta:
            continue
        cl_arr = np.array(cl)
        ranks = nd_sort(f1[cl_arr], f2[cl_arr])
        local_idx.extend(cl_arr[ranks == 1].tolist())
    return list(set(local_idx))


def _env_sel_hdm(pop, f1, f2, N, delta):
    """HDM环境选择：局部Pareto代表优先，再按全局前沿等级补充，最后汉明拥挤修剪。"""
    local_idx = _build_local_archive(pop, f1, f2, N, delta, None)
    ranks = nd_sort(f1, f2)
    P = list(local_idx)
    rest = set(range(len(f1))) - set(P)
    k = 1
    while len(P) < N:
        add = [i for i in rest if ranks[i] == k]
        P.extend(add); rest -= set(add)
        k += 1
        if k > len(f1):
            break
    P = np.array(P[:])
    if len(P) > N:
        cd = hamming_crowd(pop[P], K=3)
        order = np.argsort(-cd)         # 降序：最孤立在前
        P = P[order[:N]]
    return pop[P], f1[P], f2[P]


def _penalty_f(f1_cand, f2_cand, arc_pop, pop_cand, n_t):
    """对候选种群中靠近档案区域的个体惩罚（目标值设为最大值）。"""
    if len(arc_pop) == 0:
        return f1_cand.copy(), f2_cand.copy()
    # 汉明距离阈值：0.2 * n_t
    thresh = max(1, int(0.2 * n_t))
    hd = np.sum(pop_cand[:, None, :] != arc_pop[None, :, :], axis=2)  # (|cand|, |arc|)
    near = (hd <= thresh).any(axis=1)
    f1p, f2p = f1_cand.copy(), f2_cand.copy()
    f1p[near] = f1_cand.max(); f2p[near] = f2_cand.max()
    return f1p, f2p


def solve_hdmmode_phase(
    phase_data: PhaseData, inst: Instance,
    params: HDMMODEParams, rng: np.random.Generator,
) -> HDMMODEPhaseResult:
    n_t = phase_data.n_t
    K = 2
    N_sub = (params.N_factor * n_t) // K   # 每个子种群大小（保证整除）
    N = N_sub * K
    combo_sets = build_combo_set(phase_data, inst)

    # 初始化：随机采样后均分为K个子种群
    all_pop = np.array([random_individual(combo_sets, rng) for _ in range(N)])
    all_pop = np.array([repair(all_pop[i], combo_sets, phase_data, inst) for i in range(N)])
    xb = np.stack([decode(all_pop[i], combo_sets, inst) for i in range(N)])
    all_f1 = evaluate_f1_raw_batch(xb, phase_data, inst)
    all_f2 = evaluate_f2_raw_batch(xb)
    perm = rng.permutation(N)
    Xsub = [all_pop[perm[:N_sub]], all_pop[perm[N_sub:]]]
    F1sub = [all_f1[perm[:N_sub]], all_f1[perm[N_sub:]]]
    F2sub = [all_f2[perm[:N_sub]], all_f2[perm[N_sub:]]]

    # 各子种群局部档案（Pareto前沿）
    arcs = []
    for i in range(K):
        r = nd_sort(F1sub[i], F2sub[i])
        arcs.append(Xsub[i][r == 1])

    NND = N_sub           # 子种群1自适应规模
    conv_hist: List[float] = []
    converged = False

    for _ in range(params.G_max):
        for i in range(K):
            Xu = Xsub[i]; Fu1 = F1sub[i]; Fu2 = F2sub[i]
            # 子代：DE变异 + 修复
            Qde = de_int(Xu, combo_sets, rng, params.F_de)
            Qde = np.array([repair(Qde[j], combo_sets, phase_data, inst) for j in range(len(Qde))])
            xq = np.stack([decode(Qde[j], combo_sets, inst) for j in range(len(Qde))])
            f1q = evaluate_f1_raw_batch(xq, phase_data, inst)
            f2q = evaluate_f2_raw_batch(xq)

            cand_pop = np.vstack([Xu, Qde])
            cand_f1 = np.concatenate([Fu1, f1q])
            cand_f2 = np.concatenate([Fu2, f2q])

            if i == 0:
                # 子种群1：收敛检测 → 自适应规模
                conv_score = float(np.std(Fu1) + np.std(Fu2))
                conv_hist.append(conv_score)
                if (not converged and len(conv_hist) >= params.conv_window
                        and np.std(conv_hist[-params.conv_window:]) < params.conv_threshold):
                    converged = True
                if converged and NND > N_sub // 2:
                    NND -= 1
                target_n = NND
            else:
                target_n = N - NND
                # 子种群2：对靠近子种群1档案区域的候选施加惩罚
                cand_f1, cand_f2 = _penalty_f(cand_f1, cand_f2, arcs[0], cand_pop, n_t)

            Xsub[i], F1sub[i], F2sub[i] = _env_sel_hdm(cand_pop, cand_f1, cand_f2,
                                                         target_n, params.delta)
            # 更新档案
            r = nd_sort(F1sub[i], F2sub[i])
            arcs[i] = Xsub[i][r == 1]

    # 合并两个子种群，输出Pareto前沿
    pop_all = np.vstack(Xsub)
    f1_all = np.concatenate(F1sub); f2_all = np.concatenate(F2sub)
    pf = pareto_dominance_filter(np.column_stack([f1_all, f2_all]))
    pf_idx = np.where(pf)[0]
    return HDMMODEPhaseResult(
        t=0,
        pareto_objs=np.column_stack([f1_all[pf_idx], f2_all[pf_idx]]),
        pareto_codes=pop_all[pf_idx],
        hd=hamming_diversity(pop_all),
    )


def run_hdmmode(
    inst: Instance,
    params: Optional[HDMMODEParams] = None,
    verbose: bool = False,
) -> HDMMODEResult:
    if params is None:
        params = HDMMODEParams()
    rng = np.random.default_rng(params.seed)
    result = HDMMODEResult(instance_name=inst.name)
    for t in range(inst.T_total):
        phase = inst.phases[t]
        if verbose:
            print(f"  Phase {t+1}/{inst.T_total}")
        pr = solve_hdmmode_phase(phase, inst, params, rng)
        pr.t = t
        result.phase_results.append(pr)
    return result
