"""knowledge.py — ACK跨阶段冲突知识库（复现MATLAB knowledge_manager.m）"""
import numpy as np
from typing import List, Set, Tuple
from .problems.encoding import TargetCombos, combo_count


class KnowledgeBase:
    """
    ACK冲突知识库，跨阶段累积。
    conflict_set: set of (j, h) 0-based对，表示目标j的第h个组合（0-based）是冲突的
    elim_count[j][h]: float，目标j的第h个组合的"偏向计数"
    theta_c: 判定为冲突的阈值（默认5）
    """

    def __init__(self, theta_c: float = 5.0):
        self.theta_c = theta_c
        # key: (j, h) 0-based；value: float
        self._elim: dict[Tuple[int,int], float] = {}
        self._conflict: Set[Tuple[int,int]] = set()

    def update(self, best_c: np.ndarray, combo_sets: List[TargetCombos]) -> None:
        """
        阶段结束时调用，传入当前阶段的最优方案编码。
        对最优方案的每个 (j, h) 对，elim_count += 0.5；
        重建冲突集。
        （复现MATLAB：对最优解的选中组合累计增量0.5，近似实现）
        """
        for j, tc in enumerate(combo_sets):
            h = int(best_c[j])
            if h == 0:
                continue  # 不拦截选项不加入知识
            h0 = h - 1  # 转为0-based
            key = (j, h0)
            self._elim[key] = self._elim.get(key, 0.0) + 0.5

        # 全量重建冲突集
        self._conflict = {
            k for k, v in self._elim.items() if v >= self.theta_c
        }

    def is_conflict(self, j: int, h: int) -> bool:
        """判断目标j的第h个组合（1-based h）是否在冲突集中"""
        return (j, h - 1) in self._conflict

    def complement_set(self, j: int, tc: TargetCombos) -> List[int]:
        """
        目标j的无冲突组合1-based索引列表（不含不拦截0）
        若全为冲突，返回全集
        """
        H = combo_count(tc)
        candidates = [h for h in range(1, H + 1) if (j, h - 1) not in self._conflict]
        if not candidates:
            candidates = list(range(1, H + 1))
        return candidates


# ── 种群初始化与变异算子（依赖知识库）─────────────────────────────

def init_population(
    N: int,
    combo_sets: List[TargetCombos],
    kb: KnowledgeBase,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    知识引导混合初始化（复现MATLAB init_population.m）
    前N//2个体：每个基因位从无冲突补集采样
    后N//2个体：完全随机（0..H）
    返回 (N, n_t) int32
    """
    n_t = len(combo_sets)
    pop = np.zeros((N, n_t), dtype=np.int32)

    for idx in range(N):
        for j, tc in enumerate(combo_sets):
            H = combo_count(tc)
            if H == 0:
                pop[idx, j] = 0
                continue
            if idx < N // 2:
                comp = kb.complement_set(j, tc)
                pop[idx, j] = int(rng.choice(comp))
            else:
                pop[idx, j] = int(rng.integers(0, H + 1))  # 0..H含不拦截

    return pop


def crossover(
    pop: np.ndarray, rng: np.random.Generator, p_crossover: float = 0.8
) -> np.ndarray:
    """
    均匀交叉（复现MATLAB crossover.m）
    随机配对，每对产生1个子代，输出约N//2个子代
    """
    N, n_t = pop.shape
    idx = rng.permutation(N)
    n_pairs = N // 2
    offspring = np.empty((n_pairs, n_t), dtype=np.int32)

    for k in range(n_pairs):
        p1, p2 = pop[idx[2 * k]], pop[idx[2 * k + 1]]
        if rng.random() < p_crossover:
            mask = rng.random(n_t) < 0.5
            child = np.where(mask, p1, p2)
        else:
            child = p1.copy() if rng.random() < 0.5 else p2.copy()
        offspring[k] = child

    return offspring


def mutation(
    offspring: np.ndarray,
    combo_sets: List[TargetCombos],
    kb: KnowledgeBase,
    rng: np.random.Generator,
    p_guided: float = 0.7,
    p_mut: float = 0.05,
) -> np.ndarray:
    """
    知识引导变异（复现MATLAB mutation.m）
    对冲突基因位以p_guided概率换用无冲突组合；
    对非冲突基因位以p_mut概率完全随机变异
    """
    result = offspring.copy()
    n_ind, n_t = result.shape

    for idx in range(n_ind):
        for j, tc in enumerate(combo_sets):
            H = combo_count(tc)
            if H == 0:
                continue
            h = int(result[idx, j])

            if h > 0 and kb.is_conflict(j, h):
                # 冲突位：以p_guided概率换用无冲突组合
                if rng.random() < p_guided:
                    comp = kb.complement_set(j, tc)
                    result[idx, j] = int(rng.choice(comp))
            else:
                # 非冲突位：以p_mut概率完全随机变异
                if rng.random() < p_mut:
                    result[idx, j] = int(rng.integers(0, H + 1))

    return result
