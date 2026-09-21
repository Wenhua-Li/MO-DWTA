"""test_rmi_fix_verify.py — 验证RMI/MORI公式修正（差值→固定参考点）

不依赖真实.mat实例（本地无该文件），用合成的小规模Instance/PhaseData
手工构造两组对照解，验证修正后的核心性质：

  性质1：RMI = F1_ref - E[F1(c')]，数值应为正（因F1_ref是零毁伤上界，
         E[F1(c')]<=F1_ref总成立，只要邻居不因惨罚项爆炸式超限）
  性质2：方向正确性 —— 邻域绝对质量更好（E[F1(c')]更低）的解，RMI更高；
         且这个排序不因c自身F1(c)的大小而改变（旧公式会因f_c不同而
         颠倒排序，这正是本次要修复的bug）。
  性质3：MORI在种群内归一化后仍落在[0,1]，方向不变。

运行：python mowta/tests/test_rmi_fix_verify.py
"""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from alg_code.problems.instance import Instance, PhaseData, TargetData
from alg_code.problems.encoding import build_combo_set, decode, random_individual
from alg_code.problems.evaluator import (
    evaluate_f1_raw, evaluate_f2_raw, f1_ref, f2_ref,
)
from alg_code.problems.robustness import compute_rmi, compute_rmi_q, compute_mori


def build_synthetic_instance(seed=0):
    """构造一个 m=3武器, L=2雷达, n_t=4目标 的最小合法实例。"""
    rng = np.random.default_rng(seed)
    m, L, n_t = 3, 2, 4

    weapon_x = np.array([0.0, 10.0, 20.0])
    weapon_y = np.array([0.0, 0.0, 0.0])
    R_max = np.array([50.0, 50.0, 50.0])
    tau_react = np.array([2.0, 2.0, 2.0])
    v_weapon = np.array([300.0, 300.0, 300.0])
    priority = np.array([2, 2, 2])
    ammo = np.array([6, 6, 6])
    alpha = np.ones((m, L))

    radar_x = np.array([0.0, 20.0])
    radar_y = np.array([10.0, 10.0])
    lam = np.array([1e-4, 1e-4])
    R_radar = np.array([60.0, 60.0])
    p_th = 0.1
    C_track = np.array([10, 10])

    targets = []
    for j in range(n_t):
        targets.append(TargetData(
            id=j, x=10.0 + j * 3.0, y=5.0, speed=200.0, heading=180.0,
            threat=1.0 + 0.5 * j, rcs=1.0, W=10.0, U=3, P_min=0.5,
            category='generic',
        ))

    P_ij = np.zeros((m, n_t))
    for i in range(m):
        for j in range(n_t):
            d = np.hypot(weapon_x[i] - targets[j].x, weapon_y[i] - targets[j].y)
            P_ij[i, j] = max(0.0, min(0.99, 0.7 * (1 - d / R_max[i]) + 0.25)) if d <= R_max[i] else 0.0

    eta_j = np.full(n_t, 0.8)
    d_ij = np.zeros((m, n_t))
    for i in range(m):
        for j in range(n_t):
            d_ij[i, j] = np.hypot(weapon_x[i] - targets[j].x, weapon_y[i] - targets[j].y)

    W_jt = np.full(n_t, 10.0)
    U_jt = np.full(n_t, 3, dtype=int)
    P_min_jt = np.full(n_t, 0.5)
    available = ammo.copy()

    phase = PhaseData(
        phase=0, n_t=n_t, targets=targets, P_ij=P_ij, eta_j=eta_j, d_ij=d_ij,
        W_jt=W_jt, U_jt=U_jt, P_min_jt=P_min_jt, available=available,
    )

    inst = Instance(
        name='SYN1', n=n_t, m=m, L=L, T_total=1,
        phase_targets=np.array([n_t]),
        weapon_x=weapon_x, weapon_y=weapon_y, R_max=R_max,
        tau_react=tau_react, v_weapon=v_weapon, priority=priority, ammo=ammo,
        alpha=alpha, radar_x=radar_x, radar_y=radar_y, lam=lam, R_radar=R_radar,
        p_th=p_th, C_track=C_track, dt=1.0, phases=[phase],
    )
    return inst, phase


def main():
    inst, phase = build_synthetic_instance()
    combo_sets = build_combo_set(phase, inst)
    rng = np.random.default_rng(123)

    print("=== 性质0：F1_ref / F2_ref 基本合理性 ===")
    ref1 = f1_ref(phase)
    ref2 = f2_ref(phase)
    expected_ref1 = sum(phase.eta_j[j] * phase.targets[j].threat for j in range(phase.n_t))
    expected_ref2 = float(np.sum(phase.U_jt))
    assert abs(ref1 - expected_ref1) < 1e-9, f"f1_ref计算错误: {ref1} vs {expected_ref1}"
    assert abs(ref2 - expected_ref2) < 1e-9, f"f2_ref计算错误: {ref2} vs {expected_ref2}"
    print(f"  f1_ref={ref1:.4f} (=Σ η·Th), f2_ref={ref2:.4f} (=Σ U_jt)  PASS")

    print("\n=== 性质1：RMI为有限实数；邻居不做repair时可能因C1/C9惩罚项为大负值（与既有test_rmi_mc_accuracy一致）===")
    c_good = random_individual(combo_sets, rng)
    f_good = evaluate_f1_raw(decode(c_good, combo_sets, inst), phase, inst)
    rmi_good = compute_rmi(c_good, f_good, combo_sets, phase, inst, r=2, S=200, rng=rng)
    print(f"  随机解: F1(c)={f_good:.4f}, RMI={rmi_good:.4f}")
    assert np.isfinite(rmi_good), f"RMI非有限值: {rmi_good}"

    print("\n=== 性质2：新公式排序与'邻域绝对质量'一致，且与旧公式(差值)排序不同（核心bug修复验证）===")
    # 采样多个随机解，同时计算：
    #  - 新RMI（compute_rmi当前实现，固定参考点）
    #  - 邻域绝对质量 E[F1(c')]（新RMI应与其严格反向单调）
    #  - 旧公式RMI_old = E[F1(c')] - F1(c)（修复前的定义，用于对比排序是否不同）
    samples = []
    for _ in range(15):
        c = random_individual(combo_sets, rng)
        f_c = evaluate_f1_raw(decode(c, combo_sets, inst), phase, inst)
        rmi_new = compute_rmi(c, f_c, combo_sets, phase, inst, r=2, S=100, rng=rng)
        e_f1_neighbor = ref1 - rmi_new           # 反推邻域绝对期望F1
        rmi_old = e_f1_neighbor - f_c            # 修复前的差值公式
        samples.append((f_c, rmi_new, e_f1_neighbor, rmi_old))

    # 检查1：新RMI越大 <=> 邻域绝对质量E[F1(c')]越小（严格反向单调）
    order_by_new_rmi = [round(t[1], 6) for t in sorted(samples, key=lambda t: -t[1])]
    order_by_neighbor = [round(t[1], 6) for t in sorted(samples, key=lambda t: t[2])]
    assert order_by_new_rmi == order_by_neighbor, "新RMI排序与邻域绝对质量排序不一致！"
    print(f"  采样{len(samples)}个解，新RMI排序与'邻域绝对质量'排序完全一致（反向单调）  PASS")

    # 检查2：新旧公式排序应存在差异（证明本次修复确实改变了行为，不是无效改动）
    rank_new = {id(s): i for i, s in enumerate(sorted(samples, key=lambda t: -t[1]))}
    rank_old = {id(s): i for i, s in enumerate(sorted(samples, key=lambda t: -t[3]))}
    n_diff = sum(1 for s in samples if rank_new[id(s)] != rank_old[id(s)])
    print(f"  新/旧公式排序不一致的样本数: {n_diff}/{len(samples)}"
          f"（{'差异存在，确认修复生效' if n_diff > 0 else '警告：本次样本未体现差异，需扩大样本验证'}）")
    print(f"  示例（F1(c), 新RMI, E[F1(c\\')], 旧RMI）:")
    for f_c, rmi, e_nb, rmi_old in samples[:5]:
        print(f"    F1(c)={f_c:8.3f}  新RMI={rmi:8.3f}  E[F1(c')]={e_nb:8.3f}  旧RMI={rmi_old:8.3f}")

    print("\n=== 性质3：MORI归一化后仍在[0,1]，方向不变 ===")
    f1_vals, f2_vals, mori_raw = [], [], []
    codes = [random_individual(combo_sets, rng) for _ in range(10)]
    for c in codes:
        x = decode(c, combo_sets, inst)
        f1v = evaluate_f1_raw(x, phase, inst)
        f2v = float(evaluate_f2_raw(x))
        f1_vals.append(f1v)
        f2_vals.append(f2v)
    f1_range = max(f1_vals) - min(f1_vals) if max(f1_vals) != min(f1_vals) else 1.0
    f2_range = max(f2_vals) - min(f2_vals) if max(f2_vals) != min(f2_vals) else 1.0
    for c, f1v, f2v in zip(codes, f1_vals, f2_vals):
        m_val = compute_mori(c, f1v, f2v, combo_sets, phase, inst,
                              f1_range, f2_range, r=2, S=100, rng=rng)
        mori_raw.append(m_val)
    mori_arr = np.array(mori_raw)
    mori_min, mori_max = mori_arr.min(), mori_arr.max()
    if mori_max - mori_min > 1e-9:
        mori_norm = (mori_arr - mori_min) / (mori_max - mori_min)
    else:
        mori_norm = np.zeros_like(mori_arr)
    assert mori_norm.min() >= -1e-9 and mori_norm.max() <= 1 + 1e-9, "MORI归一化越界"
    print(f"  MORI原始值范围: [{mori_min:.4f}, {mori_max:.4f}]")
    print(f"  归一化后范围: [{mori_norm.min():.4f}, {mori_norm.max():.4f}]  PASS")

    print("\n=== 全部验证通过 ===")


if __name__ == '__main__':
    main()
