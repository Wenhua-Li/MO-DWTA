"""baselines/moead.py — MOEA/D（Tchebycheff分解）适配WTA离散问题

Reference: Q. Zhang and H. Li, MOEA/D: A multiobjective evolutionary algorithm
based on decomposition, IEEE TEC, 2007, 11(6): 712-731.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from alg_code.problems.instance import Instance, PhaseData
from alg_code.problems.encoding import (
    build_combo_set, decode, repair, combo_count, random_individual,
)
from alg_code.problems.evaluator import evaluate_f1_raw_batch, evaluate_f2_raw_batch
from alg_code.metrics import pareto_dominance_filter, hamming_diversity
from alg_code.baselines._mmea_utils import ga_offspring


@dataclass
class MOEADParams:
    N_factor: int = 5
    G_max: int = 100
    T_ratio: float = 0.1       # 邻域比例
    p_crossover: float = 0.8
    p_mut: float = 0.05
    seed: Optional[int] = None


@dataclass
class MOEADPhaseResult:
    t: int
    pareto_objs: np.ndarray    # (K, 2)
    pareto_codes: np.ndarray   # (K, n_t)
    hd: float


@dataclass
class MOEADResult:
    instance_name: str
    phase_results: List[MOEADPhaseResult] = field(default_factory=list)


def _make_child(p1, p2, combo_sets, rng, p_cx, p_mut):
    n_t = len(p1)
    child = (np.where(rng.random(n_t) < 0.5, p1, p2) if rng.random() < p_cx
             else (p1 if rng.random() < 0.5 else p2).copy()).astype(np.int32)
    for j, tc in enumerate(combo_sets):
        H = combo_count(tc)
        if H > 0 and rng.random() < p_mut:
            child[j] = int(rng.integers(0, H + 1))
    return child


def solve_moead_phase(
    phase_data: PhaseData, inst: Instance,
    params: MOEADParams, rng: np.random.Generator,
) -> MOEADPhaseResult:
    n_t = phase_data.n_t
    N = params.N_factor * n_t
    T = max(2, int(np.ceil(N * params.T_ratio)))
    combo_sets = build_combo_set(phase_data, inst)

    # 均匀权重向量（2目标单纯形）
    W = np.column_stack([np.linspace(0.0, 1.0, N), np.linspace(1.0, 0.0, N)])
    # 邻域：按权重空间欧氏距离
    dw = np.sum((W[:, None, :] - W[None, :, :]) ** 2, axis=2)
    B = np.argsort(dw, axis=1)[:, :T]  # (N, T)

    pop = np.array([random_individual(combo_sets, rng) for _ in range(N)])
    pop = np.array([repair(pop[i], combo_sets, phase_data, inst) for i in range(N)])
    xb = np.stack([decode(pop[i], combo_sets, inst) for i in range(N)])
    f1 = evaluate_f1_raw_batch(xb, phase_data, inst)
    f2 = evaluate_f2_raw_batch(xb)
    z = np.array([f1.min(), f2.min()])   # 理想点

    for _ in range(params.G_max):
        for i in range(N):
            nbrs = B[i]
            pa, pb = rng.choice(nbrs, 2, replace=False)
            child = _make_child(pop[pa], pop[pb], combo_sets, rng,
                                params.p_crossover, params.p_mut)
            child = repair(child, combo_sets, phase_data, inst)
            xc = decode(child, combo_sets, inst)
            f1c = float(evaluate_f1_raw_batch(xc[None], phase_data, inst)[0])
            f2c = float(evaluate_f2_raw_batch(xc[None])[0])
            z[0] = min(z[0], f1c); z[1] = min(z[1], f2c)

            # Tchebycheff更新邻域
            g_old = np.maximum(W[nbrs, 0] * np.abs(f1[nbrs] - z[0]),
                               W[nbrs, 1] * np.abs(f2[nbrs] - z[1]))
            g_new = np.maximum(W[nbrs, 0] * abs(f1c - z[0]),
                               W[nbrs, 1] * abs(f2c - z[1]))
            upd = g_new < g_old
            pop[nbrs[upd]] = child
            f1[nbrs[upd]] = f1c
            f2[nbrs[upd]] = f2c

    pf = pareto_dominance_filter(np.column_stack([f1, f2]))
    pf_idx = np.where(pf)[0]
    return MOEADPhaseResult(
        t=0,
        pareto_objs=np.column_stack([f1[pf_idx], f2[pf_idx]]),
        pareto_codes=pop[pf_idx],
        hd=hamming_diversity(pop),
    )


def run_moead(
    inst: Instance,
    params: Optional[MOEADParams] = None,
    verbose: bool = False,
) -> MOEADResult:
    if params is None:
        params = MOEADParams()
    rng = np.random.default_rng(params.seed)
    result = MOEADResult(instance_name=inst.name)
    for t in range(inst.T_total):
        phase = inst.phases[t]
        if verbose:
            print(f"  Phase {t+1}/{inst.T_total}")
        pr = solve_moead_phase(phase, inst, params, rng)
        pr.t = t
        result.phase_results.append(pr)
    return result
