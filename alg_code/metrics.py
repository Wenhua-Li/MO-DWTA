"""metrics.py — 多目标评价指标（HV/IGD/HD/PSE/PSR/PSS）"""
import numpy as np
from typing import List, Optional
from .problems.encoding import hamming_distance
from .problems.evaluator import f1_max_from_residual


# ── HV (Hypervolume) ───────────────────────────────────────────────────

def hypervolume(pareto_set: np.ndarray, ref_point: np.ndarray) -> float:
    """
    2D Hypervolume（WFG算法，精确）
    pareto_set: (N, 2) 归一化目标向量，越小越好
    ref_point:  (2,) 参考点，应大于pareto_set的nadir点
    返回: HV值（越大越好）
    """
    if len(pareto_set) == 0:
        return 0.0
    # 非支配排序预处理（确保输入是非支配的）
    ps = pareto_set[pareto_dominance_filter(pareto_set)]
    if len(ps) == 0:
        return 0.0
    # 按第一维升序
    order = np.argsort(ps[:, 0])
    ps = ps[order]
    # WFG 2D递归
    hv = 0.0
    prev_y = ref_point[1]
    for i in range(len(ps)):
        x, y = ps[i]
        if x >= ref_point[0] or y >= ref_point[1]:
            continue
        width = ref_point[0] - x
        height = prev_y - y
        hv += width * height
        prev_y = y
    return max(0.0, hv)


def pareto_dominance_filter(objs: np.ndarray) -> np.ndarray:
    """返回非支配解的bool mask（标准Pareto支配，越小越好）"""
    N = len(objs)
    dominated = np.zeros(N, dtype=bool)
    for i in range(N):
        if dominated[i]:
            continue
        for j in range(i + 1, N):
            if dominated[j]:
                continue
            # i支配j: i <= j 且至少一维严格小
            if (objs[i] <= objs[j]).all() and (objs[i] < objs[j]).any():
                dominated[j] = True
            # j支配i
            elif (objs[j] <= objs[i]).all() and (objs[j] < objs[i]).any():
                dominated[i] = True
                break
    return ~dominated


# ── IGD (Inverted Generational Distance) ───────────────────────────────

def igd(pareto_set: np.ndarray, reference_front: np.ndarray) -> float:
    """
    IGD = mean( min_distance(r, pareto_set) for r in reference_front )
    越小越好（0=完美逼近）
    pareto_set: (N, 2)
    reference_front: (M, 2)
    """
    if len(pareto_set) == 0 or len(reference_front) == 0:
        return np.inf
    dists = []
    for ref in reference_front:
        d = np.linalg.norm(pareto_set - ref, axis=1).min()
        dists.append(d)
    return float(np.mean(dists))


# ── HD (Hamming Diversity) ──────────────────────────────────────────────

def hamming_diversity(pop: np.ndarray) -> float:
    """
    HD = (1 / N(N-1)) * Σ_{i≠j} d_TC(c_i, c_j)
    pop: (N, n_t) int32编码
    越大越好（决策空间多样性高）
    """
    N = len(pop)
    if N <= 1:
        return 0.0
    total = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            total += hamming_distance(pop[i], pop[j])
    return total / (N * (N - 1) / 2)


# ── 后选择指标（PSE/PSR/PSS）────────────────────────────────────────────

def pse_metric(selected_f1_res: List[float], phase_data_list) -> float:
    """
    PSE = 所有阶段后选择方案的F1加权拦截成功概率均值
    （转换：F1_max = Σ η·Th - F1_res）
    越大越好
    """
    f1_maxs = [
        f1_max_from_residual(f1_res, pd)
        for f1_res, pd in zip(selected_f1_res, phase_data_list)
    ]
    return float(np.mean(f1_maxs))


def psr_metric(selected_rmi: List[float]) -> float:
    """
    PSR = 所有阶段后选择方案的RMI均值
    越大越好（鲁棒性强）
    """
    return float(np.mean(selected_rmi))


def pss_metric(selected_encodings: List[np.ndarray]) -> float:
    """
    PSS = 相邻阶段后选择方案的汉明距离均值
    越小越好（切换代价小、时序一致）
    """
    if len(selected_encodings) <= 1:
        return 0.0
    dists = [
        hamming_distance(selected_encodings[t], selected_encodings[t + 1])
        for t in range(len(selected_encodings) - 1)
    ]
    return float(np.mean(dists))


# ── 辅助：计算参考点 ────────────────────────────────────────────────────

def compute_reference_point(all_objs: List[np.ndarray], factor: float = 1.1) -> np.ndarray:
    """
    从多个算法的合并目标集中计算参考点（nadir * factor）
    all_objs: list of (N_i, 2) 归一化目标向量
    """
    combined = np.vstack(all_objs)
    nadir = combined.max(axis=0)
    return nadir * factor


# ── 周期内归一化 ────────────────────────────────────────────────────────

def normalize_objectives(f1_res: np.ndarray, f2: np.ndarray, phase_data) -> np.ndarray:
    """
    周期内min-max归一化（阶段1及后续使用）
    f1_res: (N,) 残余威胁
    f2:     (N,) 弹药消耗
    phase_data: PhaseData（用于F1上界Σ η·Th）
    返回: (N, 2) 归一化目标向量，[0,1]范围，越小越好
    """
    # F1: 先转为成功概率（越大越好），再归一化到[0,1]（大的映射到0）
    f1_max_upper = sum(
        phase_data.eta_j[j] * phase_data.targets[j].threat
        for j in range(phase_data.n_t)
    )
    f1_success = f1_max_upper - f1_res  # 成功概率（越大越好）
    f1_norm = 1.0 - (f1_success / f1_max_upper if f1_max_upper > 0 else 0)  # 转为越小越好

    # F2: min-max归一化（越小越好直接归一化）
    f2_min, f2_max = f2.min(), f2.max()
    if f2_max - f2_min > 1e-9:
        f2_norm = (f2 - f2_min) / (f2_max - f2_min)
    else:
        f2_norm = np.zeros_like(f2)

    return np.column_stack([f1_norm, f2_norm])
