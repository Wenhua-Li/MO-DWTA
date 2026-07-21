"""encoding.py — TC-FCE编码、约束预处理5步、C1/C9贪心修复"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from .instance import Instance, PhaseData


# ── 数据结构 ──────────────────────────────────────────────────────────

@dataclass
class Combo:
    """单个可行交战组合（均0-based索引）"""
    weapon: int   # 武器平台索引（0-based），-1表示不拦截
    radar: int    # 雷达索引（0-based），-1表示不拦截
    shots: int    # 弹药数，0表示不拦截


NO_INTERCEPT = Combo(-1, -1, 0)


@dataclass
class TargetCombos:
    """目标j的所有可行交战组合"""
    target_idx: int           # 目标索引（0-based）
    combos: List[Combo]       # 所有组合，最后一个始终为NO_INTERCEPT
    # H_jt = len(combos) - 1，即有效组合数（不含不拦截）


def combo_count(tc: TargetCombos) -> int:
    """有效组合数（不含不拦截）"""
    return len(tc.combos) - 1


# 编码：c[j] = 0 → 不拦截；c[j] = h（1-based）→ tc.combos[h-1]
# 实现上统一用0-based存储，解码时 c[j]-1 即为combos下标，0表示不拦截


# ── 约束预处理 ────────────────────────────────────────────────────────

def build_combo_set(phase_data: PhaseData, inst: Instance) -> List[TargetCombos]:
    """
    约束预处理5步，构造每个目标的可行交战组合集（对应MATLAB constraint_preprocess.m）
    返回长度n_t的TargetCombos列表（0-based）
    """
    m, L = inst.m, inst.L
    n_t = phase_data.n_t
    targets = phase_data.targets
    avail = phase_data.available   # (m,) 当前可用弹药
    result: List[TargetCombos] = []

    for j, tgt in enumerate(targets):
        combos: List[Combo] = []

        # Step 1: 有效探测雷达子集 B_jt（C6, C7）
        B_jt = []
        for k in range(L):
            d_kj = np.hypot(inst.radar_x[k] - tgt.x, inst.radar_y[k] - tgt.y)
            if d_kj > inst.R_radar[k]:
                continue
            sigma = max(tgt.rcs, 1e-6)
            p_kj = max(0.05, float(np.exp(-inst.lam[k] * d_kj**4 / sigma)))
            if p_kj >= inst.p_th:
                B_jt.append(k)

        if not B_jt:
            # 不可探测目标：仅不拦截选项
            result.append(TargetCombos(j, [NO_INTERCEPT]))
            continue

        # Step 2 & 3: 可行(i,k)对 + 枚举弹药数 s（C2,C4,C5,C8,C12,C3）
        threat_high = tgt.threat > 0.7

        for k in B_jt:
            for i in range(m):
                # C8: 武器-雷达协同
                if inst.alpha[i, k] == 0:
                    continue
                # C2: 弹药可用
                if avail[i] <= 0:
                    continue
                # C4: 射程
                d_ij = float(phase_data.d_ij[i, j])
                if d_ij > inst.R_max[i]:
                    continue
                # C5: 时效窗口
                if inst.v_weapon[i] > 0:
                    t_fly = inst.tau_react[i] + d_ij / inst.v_weapon[i]
                else:
                    t_fly = inst.tau_react[i]
                if t_fly > tgt.W:
                    continue
                # C12: 武器优先级（高威胁目标需高优先级武器）
                if threat_high and inst.priority[i] < 2:
                    continue

                # Step 3: 枚举弹药数 s（C3：不超过目标上限且不超过可用弹药）
                max_s = min(int(avail[i]), int(tgt.U))
                for s in range(1, max_s + 1):
                    combos.append(Combo(i, k, s))

        # Step 4 & 5: C9 C11 剪枝（简化：不预剪，由修复处理）

        # 末尾追加不拦截选项
        combos.append(NO_INTERCEPT)
        result.append(TargetCombos(j, combos))

    return result


# ── 编码与解码 ────────────────────────────────────────────────────────

def decode(c: np.ndarray, combo_sets: List[TargetCombos], inst: Instance) -> np.ndarray:
    """
    TC-FCE解码：编码向量 → x_ijkt (m, n_t, L) int数组
    c[j] = 0       → 不拦截（NO_INTERCEPT）
    c[j] = h ≥ 1  → combo_sets[j].combos[h-1]
    """
    n_t = len(combo_sets)
    m, L = inst.m, inst.L
    x = np.zeros((m, n_t, L), dtype=np.int32)

    for j in range(n_t):
        h = int(c[j])
        if h == 0:
            continue
        tc = combo_sets[j]
        if h - 1 >= len(tc.combos):
            continue  # 越界保护：视为不拦截
        combo = tc.combos[h - 1]
        if combo.weapon < 0:
            continue  # NO_INTERCEPT
        x[combo.weapon, j, combo.radar] = combo.shots

    return x


def random_individual(combo_sets: List[TargetCombos], rng: np.random.Generator) -> np.ndarray:
    """随机生成一个合法编码（含不拦截选项）"""
    n_t = len(combo_sets)
    c = np.zeros(n_t, dtype=np.int32)
    for j in range(n_t):
        H = combo_count(combo_sets[j])  # 有效组合数（不含末尾不拦截）
        if H > 0:
            # 0 = 不拦截，1..H = 有效组合
            c[j] = rng.integers(0, H + 1)
    return c


# ── C1/C9 贪心修复 ────────────────────────────────────────────────────

def repair(c: np.ndarray, combo_sets: List[TargetCombos],
           phase_data: PhaseData, inst: Instance) -> np.ndarray:
    """
    贪心修复C1（武器弹药超限）和C9（雷达通道超限）。
    复现 MATLAB repair_constraints.m：
      按威胁降序处理，尝试换用合法组合；若无则设为不拦截。
    修改在副本上进行，返回修复后的编码。
    """
    c = c.copy()
    n_t = len(combo_sets)
    targets = phase_data.targets
    avail = phase_data.available.astype(int).copy()
    C_track = inst.C_track.astype(int).copy()
    m, L = inst.m, inst.L

    # 统计当前用量
    weapon_used = np.zeros(m, dtype=int)
    radar_used = np.zeros(L, dtype=int)
    for j in range(n_t):
        h = int(c[j])
        if h == 0:
            continue
        combo = combo_sets[j].combos[h - 1]
        if combo.weapon < 0:
            continue
        weapon_used[combo.weapon] += combo.shots
        radar_used[combo.radar] += combo.shots

    # 若无违约，直接返回
    if (weapon_used <= avail).all() and (radar_used <= C_track).all():
        return c

    # 按威胁降序处理
    order = sorted(range(n_t), key=lambda j: targets[j].threat, reverse=True)

    for j in order:
        h = int(c[j])
        if h == 0:
            continue
        combo = combo_sets[j].combos[h - 1]
        if combo.weapon < 0:
            continue
        wi, rk, s = combo.weapon, combo.radar, combo.shots

        if weapon_used[wi] > avail[wi] or radar_used[rk] > C_track[rk]:
            # 尝试找替代组合
            weapon_used[wi] -= s
            radar_used[rk] -= s

            found = False
            for h2, alt in enumerate(combo_sets[j].combos[:-1], start=1):  # 不含末尾不拦截
                if alt.weapon < 0:
                    continue
                wi2, rk2, s2 = alt.weapon, alt.radar, alt.shots
                if (weapon_used[wi2] + s2 <= avail[wi2] and
                        radar_used[rk2] + s2 <= C_track[rk2]):
                    c[j] = h2
                    weapon_used[wi2] += s2
                    radar_used[rk2] += s2
                    found = True
                    break

            if not found:
                c[j] = 0  # 不拦截

    return c


# ── 汉明距离 ──────────────────────────────────────────────────────────

def hamming_distance(c1: np.ndarray, c2: np.ndarray) -> int:
    """d_TC(c1, c2) = Σ_j 1[c1[j] != c2[j]]（无加权）"""
    return int(np.sum(c1 != c2))


def hamming_matrix(pop: np.ndarray) -> np.ndarray:
    """
    计算种群两两汉明距离矩阵（N, N），对角为0。
    pop: (N, n_t) int32
    """
    N = len(pop)
    # 利用广播：pop[i] != pop[j] 对每位比较后求和
    # 内存高效写法：逐行
    D = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        # 一次广播比较 pop[i] vs 所有行
        D[i] = (pop[i] != pop).sum(axis=1)
    return D
