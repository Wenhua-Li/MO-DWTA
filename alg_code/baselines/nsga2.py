"""baselines/nsga2.py — 标准NSGA-II适配WTA离散问题"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from alg_code.problems.instance import Instance, PhaseData
from alg_code.problems.encoding import (
    build_combo_set, decode, repair, hamming_distance,
    TargetCombos, combo_count, random_individual,
)
from alg_code.problems.evaluator import evaluate_f1_raw_batch, evaluate_f2_raw_batch
from alg_code.metrics import pareto_dominance_filter, hamming_diversity


@dataclass
class NSGA2Params:
    N_factor: int = 5
    G_max: int = 100
    p_crossover: float = 0.8
    p_mut: float = 0.05
    seed: Optional[int] = None


@dataclass
class NSGA2PhaseResult:
    t: int
    pareto_objs: np.ndarray    # (K, 2) [f1_res, f2]
    pareto_codes: np.ndarray   # (K, n_t)
    hd: float                  # 汉明多样性（种群级别）


@dataclass
class NSGA2Result:
    instance_name: str
    phase_results: List[NSGA2PhaseResult] = field(default_factory=list)


# ── NSGA-II非支配排序 ────────────────────────────────────────────────

def fast_nondom_sort(f1: np.ndarray, f2: np.ndarray) -> List[List[int]]:
    """标准快速非支配排序（Deb 2002）"""
    N = len(f1)
    dominated_by = [[] for _ in range(N)]
    dom_count = np.zeros(N, dtype=int)
    fronts: List[List[int]] = []
    current = []

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # i支配j
            if (f1[i] <= f1[j] and f2[i] <= f2[j]) and (f1[i] < f1[j] or f2[i] < f2[j]):
                dominated_by[i].append(j)
                dom_count[j] += 1
        if dom_count[i] == 0:
            current.append(i)

    while current:
        fronts.append(current)
        nxt = []
        for i in current:
            for j in dominated_by[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    nxt.append(j)
        current = nxt
    return fronts


def crowding_distance(indices: List[int], f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    n = len(indices)
    if n <= 2:
        return np.full(n, np.inf)
    idx = np.array(indices)
    cd = np.zeros(n)
    for vals in [f1[idx], f2[idx]]:
        order = np.argsort(vals)
        cd[order[0]] = np.inf
        cd[order[-1]] = np.inf
        r = vals[order[-1]] - vals[order[0]]
        if r < 1e-12:
            continue
        for k in range(1, n - 1):
            cd[order[k]] += (vals[order[k+1]] - vals[order[k-1]]) / r
    return cd


def nsga2_select(pop: np.ndarray, f1: np.ndarray, f2: np.ndarray, N: int):
    """NSGA-II环境选择（按前沿+拥挤距离）"""
    fronts = fast_nondom_sort(f1, f2)
    new_idx: List[int] = []
    for front in fronts:
        if len(new_idx) + len(front) <= N:
            new_idx.extend(front)
        else:
            need = N - len(new_idx)
            cd = crowding_distance(front, f1, f2)
            order = np.argsort(-cd)
            new_idx.extend([front[k] for k in order[:need]])
            break
    sel = np.array(new_idx[:N])
    return pop[sel], f1[sel], f2[sel]


# ── 简单均匀交叉+变异（无知识库）─────────────────────────────────────

def crossover_simple(pop: np.ndarray, rng: np.random.Generator, p_cx: float = 0.8) -> np.ndarray:
    N, n_t = pop.shape
    idx = rng.permutation(N)
    offspring = []
    for k in range(N // 2):
        p1, p2 = pop[idx[2*k]], pop[idx[2*k+1]]
        if rng.random() < p_cx:
            mask = rng.random(n_t) < 0.5
            child = np.where(mask, p1, p2)
        else:
            child = p1.copy() if rng.random() < 0.5 else p2.copy()
        offspring.append(child)
    return np.array(offspring, dtype=np.int32)


def mutation_simple(
    off: np.ndarray, combo_sets: List[TargetCombos],
    rng: np.random.Generator, p_mut: float = 0.05
) -> np.ndarray:
    result = off.copy()
    for i in range(len(result)):
        for j, tc in enumerate(combo_sets):
            H = combo_count(tc)
            if H > 0 and rng.random() < p_mut:
                result[i, j] = int(rng.integers(0, H + 1))
    return result


# ── 主求解器 ──────────────────────────────────────────────────────────

def solve_nsga2_phase(
    phase_data: PhaseData, inst: Instance,
    params: NSGA2Params, rng: np.random.Generator,
) -> NSGA2PhaseResult:
    n_t = phase_data.n_t
    N = params.N_factor * n_t
    combo_sets = build_combo_set(phase_data, inst)

    # 随机初始化
    pop = np.array([random_individual(combo_sets, rng) for _ in range(N)])
    pop = np.array([repair(pop[i], combo_sets, phase_data, inst) for i in range(N)])

    x_batch = np.stack([decode(pop[i], combo_sets, inst) for i in range(N)])
    f1 = evaluate_f1_raw_batch(x_batch, phase_data, inst)
    f2 = evaluate_f2_raw_batch(x_batch)

    for g in range(params.G_max):
        Q = crossover_simple(pop, rng, params.p_crossover)
        Q = mutation_simple(Q, combo_sets, rng, params.p_mut)
        Q = np.array([repair(Q[i], combo_sets, phase_data, inst) for i in range(len(Q))])
        x_q = np.stack([decode(Q[i], combo_sets, inst) for i in range(len(Q))])
        f1_q = evaluate_f1_raw_batch(x_q, phase_data, inst)
        f2_q = evaluate_f2_raw_batch(x_q)

        combined = np.vstack([pop, Q])
        f1_c = np.concatenate([f1, f1_q])
        f2_c = np.concatenate([f2, f2_q])
        pop, f1, f2 = nsga2_select(combined, f1_c, f2_c, N)

    # 返回最终第一前沿
    fronts = fast_nondom_sort(f1, f2)
    pf_idx = np.array(fronts[0])
    hd = hamming_diversity(pop)

    return NSGA2PhaseResult(
        t=0,
        pareto_objs=np.column_stack([f1[pf_idx], f2[pf_idx]]),
        pareto_codes=pop[pf_idx],
        hd=hd,
    )


def run_nsga2(
    inst: Instance,
    params: Optional[NSGA2Params] = None,
    verbose: bool = False,
) -> NSGA2Result:
    if params is None:
        params = NSGA2Params()
    rng = np.random.default_rng(params.seed)
    result = NSGA2Result(instance_name=inst.name)

    for t in range(inst.T_total):
        phase = inst.phases[t]
        if verbose:
            print(f"  Phase {t+1}/{inst.T_total}")
        pr = solve_nsga2_phase(phase, inst, params, rng)
        pr.t = t
        result.phase_results.append(pr)

    return result
