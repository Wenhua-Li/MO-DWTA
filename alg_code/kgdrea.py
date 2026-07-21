"""kgdrea.py — 单目标KG-DREA（全新Python重写，行为参照Paper2 MATLAB）

算法：KG-DREA（Knowledge-Guided Dynamic Rolling Evolutionary Algorithm）
用途：①阶段0等价性验证基准；②论文对比中的"单目标下界"基线
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .problems.instance import Instance, PhaseData, load_instance
from .problems.encoding import (
    build_combo_set, decode, repair, hamming_distance, hamming_matrix,
    TargetCombos, combo_count, random_individual,
)
from .problems.evaluator import evaluate_f1_raw, evaluate_f1_raw_batch
from .problems.robustness import compute_rmi, _decode_fast, adaptive_sample_size
from .knowledge import KnowledgeBase, init_population, crossover, mutation


@dataclass
class KGDREAParams:
    N_factor: int = 5          # 种群规模 = N_factor * n_t
    G_max: int = 100           # 最大进化代数
    delta_factor: float = 0.2  # 小生境半径 = ceil(delta_factor * n_t)
    epsilon_rmi: float = 0.01  # 近似最优解集阈值
    r: int = 2                 # RMI邻域半径
    S: int = -1                # -1 表示自动自适应；正整数为 override
    eps_rel: float = 0.15      # Hoeffding 相对精度
    delta_prob: float = 0.10   # 允许失败概率
    p_crossover: float = 0.8
    p_mut: float = 0.05
    p_guided: float = 0.7
    theta_c: float = 5.0
    seed: Optional[int] = None


@dataclass
class PhaseResult:
    t: int                    # 阶段编号（0-based）
    best_c: np.ndarray        # 最优方案编码
    best_f: float             # 最优方案F1_res（残余威胁，越小越好）
    best_rmi: float           # 最优方案RMI
    fitness_history: List[float] = field(default_factory=list)  # 每代最优适应度


@dataclass
class KGDREAResult:
    instance_name: str
    phase_results: List[PhaseResult] = field(default_factory=list)
    total_f: float = 0.0       # 所有阶段f1_res之和


# ── 环境选择：汉明距离小生境 ─────────────────────────────────────────

def niche_selection(
    pop: np.ndarray,
    fitness: np.ndarray,
    N: int,
    delta: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    汉明距离小生境环保选择（复现MATLAB niche_selection.m）
    按适应度升序扫描候选池；遇到与已选集中有个体距离<=delta时，
    与第一个邻近个体比较：若候选更优则替换，否则丢弃。

    pop:     (R, n_t) int32，候选池
    fitness: (R,) float，候选池适应度（越小越好）
    N:       目标种群规模
    delta:   小生境半径
    返回: (selected_pop (N,n_t), selected_fit (N,))
    """
    order = np.argsort(fitness)  # 升序
    selected: List[int] = []

    for idx in order:
        if len(selected) >= N:
            break
        diverse = True
        replace_at = -1
        for s_idx, s in enumerate(selected):
            d = hamming_distance(pop[idx], pop[s])
            if d <= delta:
                diverse = False
                if fitness[idx] < fitness[s]:
                    replace_at = s_idx
                break  # 只与第一个邻近个体比较（复现MATLAB break）
        if diverse:
            selected.append(idx)
        elif replace_at >= 0:
            selected[replace_at] = idx

    # 若不足N，从候选池补充（随机，可含重复，复现MATLAB行为）
    if len(selected) < N:
        remaining = [i for i in range(len(pop)) if i not in set(selected)]
        need = N - len(selected)
        if remaining:
            extra = np.random.choice(remaining, size=min(need, len(remaining)), replace=False)
            selected.extend(extra.tolist())
        # 仍不足则重复填充（极端情况）
        while len(selected) < N:
            selected.append(selected[-1])

    sel = np.array(selected[:N])
    return pop[sel], fitness[sel]


# ── 主求解器 ──────────────────────────────────────────────────────────

def solve_single_phase(
    phase_data: PhaseData,
    inst: Instance,
    kb: KnowledgeBase,
    params: KGDREAParams,
    rng: np.random.Generator,
    verbose: bool = False,
) -> tuple[np.ndarray, float, float, List[float]]:
    """
    对单个决策周期运行KG-DREA进化循环。
    返回 (best_c, best_f, best_rmi, fitness_history)
    """
    n_t = phase_data.n_t
    N = params.N_factor * n_t
    delta = max(1, int(np.ceil(params.delta_factor * n_t)))

    # 约束预处理
    combo_sets = build_combo_set(phase_data, inst)

    # 知识引导初始化
    pop = init_population(N, combo_sets, kb, rng)
    # 修复 + 批量评估
    pop = np.array([repair(pop[i], combo_sets, phase_data, inst) for i in range(N)])
    x_batch = np.stack([decode(pop[i], combo_sets, inst) for i in range(N)])
    fit = evaluate_f1_raw_batch(x_batch, phase_data, inst)

    history: List[float] = [float(fit.min())]

    # 进化主循环
    for g in range(params.G_max):
        # 交叉 + 变异 + 修复
        Q = crossover(pop, rng, params.p_crossover)
        Q = mutation(Q, combo_sets, kb, rng, params.p_guided, params.p_mut)
        Q = np.array([repair(Q[i], combo_sets, phase_data, inst) for i in range(len(Q))])
        x_q = np.stack([decode(Q[i], combo_sets, inst) for i in range(len(Q))])
        fit_q = evaluate_f1_raw_batch(x_q, phase_data, inst)

        # 合并父代+子代
        combined = np.vstack([pop, Q])
        combined_fit = np.concatenate([fit, fit_q])

        # 小生境环保选择
        pop, fit = niche_selection(combined, combined_fit, N, delta)
        history.append(float(fit.min()))

        if verbose and (g + 1) % 20 == 0:
            print(f"    gen {g+1}: best_f={fit.min():.4f}")

    # 近似最优解集 + RMI优选
    f_min = float(fit.min())
    eps_threshold = f_min * (1.0 + params.epsilon_rmi)
    elite_mask = fit <= eps_threshold
    elite_idx = np.where(elite_mask)[0]

    delta_niche = max(1.0, params.delta_factor * n_t)
    # 自适应采样数（一次计算，复用于所有精英解）
    if params.S > 0:
        S_eval = params.S
    else:
        S_eval_raw = adaptive_sample_size(
            n_t, combo_sets, params.r, delta_niche,
            params.eps_rel, params.delta_prob,
        )
        S_eval = S_eval_raw if S_eval_raw > 0 else 30  # exact路径兜底

    best_c = pop[elite_idx[0]]
    best_f = fit[elite_idx[0]]
    best_rmi = -np.inf

    for idx in elite_idx:
        c = pop[idx]
        f_c = float(fit[idx])
        rmi = compute_rmi(c, f_c, combo_sets, phase_data, inst,
                          r=params.r, S=S_eval, rng=rng,
                          delta=delta_niche, eps_rel=params.eps_rel, delta_prob=params.delta_prob)
        if rmi > best_rmi:
            best_rmi = rmi
            best_c = c
            best_f = f_c

    return best_c, best_f, float(best_rmi), history


def run_kgdrea(
    inst: Instance,
    params: Optional[KGDREAParams] = None,
    verbose: bool = False,
) -> KGDREAResult:
    """
    完整多阶段KG-DREA运行入口。
    """
    if params is None:
        params = KGDREAParams()
    rng = np.random.default_rng(params.seed)
    kb = KnowledgeBase(theta_c=params.theta_c)
    result = KGDREAResult(instance_name=inst.name)
    total_f = 0.0

    for t in range(inst.T_total):
        phase_data = inst.phases[t]
        if verbose:
            print(f"  Phase {t+1}/{inst.T_total}, n_t={phase_data.n_t}")

        best_c, best_f, best_rmi, history = solve_single_phase(
            phase_data, inst, kb, params, rng, verbose=verbose
        )

        # 知识库更新（使用当前阶段的combo_sets）
        combo_sets = build_combo_set(phase_data, inst)
        kb.update(best_c, combo_sets)

        pr = PhaseResult(t=t, best_c=best_c, best_f=best_f,
                         best_rmi=best_rmi, fitness_history=history)
        result.phase_results.append(pr)
        total_f += best_f

    result.total_f = total_f
    return result


# ── 多次运行入口（用于等价性验证）─────────────────────────────────

def run_multiple(
    inst: Instance,
    n_runs: int = 30,
    params: Optional[KGDREAParams] = None,
    verbose: bool = False,
) -> List[float]:
    """
    对同一实例独立运行n_runs次，返回每次total_f（全阶段残余威胁之和）列表
    """
    results = []
    base_seed = (params.seed if params and params.seed is not None else 42)
    for run in range(n_runs):
        p = KGDREAParams(**(params.__dict__ if params else {}))
        p.seed = base_seed + run
        r = run_kgdrea(inst, p, verbose=False)
        results.append(r.total_f)
        if verbose:
            print(f"Run {run+1}/{n_runs}: total_f={r.total_f:.4f}")
    return results
