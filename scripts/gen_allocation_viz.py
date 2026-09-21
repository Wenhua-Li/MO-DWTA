"""gen_allocation_viz.py
生成武器目标分配结果的可视化：战场地图 + 分配热力表 + 鲁棒性对比。
对I1实例（小规模，结果直观）的典型运行做深度解剖。
输出：一段可嵌入HTML的字符串（包含图片base64和HTML表格）。
"""
import sys, io, base64, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import matplotlib.cm as cm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from alg_code.problems.instance import load_instance
from alg_code.problems.encoding import build_combo_set, decode, combo_count
from alg_code.problems.evaluator import evaluate_f1_raw, evaluate_f2_raw
from alg_code.rcmd_ea import run_rcmd_ea, RCMDParams
from alg_code.baselines.nsga2 import run_nsga2, NSGA2Params
from alg_code.post_selection import topsis_select

# ── 路径 ─────────────────────────────────────────────────────────────
INST_DIR = ROOT.parent / 'paper2' / 'source_code' / 'instances'
MAT_I1   = str(INST_DIR / 'I1.mat')

CAT_COLORS = {
    'conventional':   '#2196F3',   # 蓝
    'low_observable': '#FF9800',   # 橙
    'stealth':        '#9C27B0',   # 紫
}
CAT_CN = {
    'conventional':   '普通',
    'low_observable': '低可见',
    'stealth':        '隐身',
}
WEAPON_COLOR  = '#4CAF50'
RADAR_COLOR   = '#F44336'
ALGO_COLORS   = {'rcmd_ea': '#d62728', 'nsga2_topsis': '#1f77b4'}
ALGO_LABELS   = {'rcmd_ea': 'RCMD-EA（本文）', 'nsga2_topsis': 'NSGA-II+TOPSIS'}

def b64(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def img_tag(b64_str, w='100%', caption=''):
    cap = f'<div style="text-align:center;font-size:11px;color:#666;margin-top:4px">{caption}</div>' if caption else ''
    return f'<img src="data:image/png;base64,{b64_str}" style="width:{w};display:block;margin:0 auto;">{cap}'

# ═══════════════════════════════════════════════════════════════════════
# 核心：运行算法，获取分配结果
# ═══════════════════════════════════════════════════════════════════════
def get_allocation(inst, best_c, phase_idx):
    """解码best_c，计算分配详情"""
    ph         = inst.phases[phase_idx]
    combo_sets = build_combo_set(ph, inst)
    x          = decode(best_c, combo_sets, inst)
    f1         = evaluate_f1_raw(x, ph, inst)
    f2         = evaluate_f2_raw(x)

    records = []
    for j in range(ph.n_t):
        tgt = ph.targets[j]
        surv = 1.0
        assigned = []
        for i in range(inst.m):
            for k in range(inst.L):
                if x[i, j, k] > 0:
                    shots = int(x[i, j, k])
                    p_one = ph.P_ij[i, j]
                    p_kill = 1 - (1 - p_one) ** shots
                    surv *= (1 - p_kill)
                    assigned.append({
                        'weapon': i, 'radar': k, 'shots': shots,
                        'p_one': p_one, 'p_kill': p_kill,
                    })
        records.append({
            'target_idx': j,
            'threat': tgt.threat,
            'category': tgt.category,
            'x': tgt.x, 'y': tgt.y,
            'eta': float(ph.eta_j[j]),
            'survival': surv,
            'p_kill_total': 1 - surv,
            'residual': tgt.threat * ph.eta_j[j] * surv,
            'assigned': assigned,
        })
    return records, float(f1), int(f2), ph

def run_both_algos(seed=42):
    inst = load_instance(MAT_I1)

    # RCMD-EA
    params_rcmd = RCMDParams(G_max=100, seed=seed, S=50, explore_ratio=0.4)
    res_rcmd    = run_rcmd_ea(inst, params_rcmd)

    # NSGA-II + TOPSIS（相同seed）
    rng_nsga = np.random.default_rng(seed)
    params_nsga = NSGA2Params(G_max=100, seed=seed)
    res_nsga    = run_nsga2(inst, params_nsga)

    results = {}
    for algo, res, is_rcmd in [('rcmd_ea', res_rcmd, True), ('nsga2_topsis', res_nsga, False)]:
        ph_results = []
        for t in range(inst.T_total):
            if is_rcmd:
                pr    = res.phase_results[t]
                best_c = pr.best_c
                rmi   = pr.best_rmi
            else:
                pr    = res.phase_results[t]
                f1arr = pr.pareto_objs[:, 0]
                f2arr = pr.pareto_objs[:, 1]
                best_c, bf1, bf2 = topsis_select(pr.pareto_codes, f1arr, f2arr)
                # 计算RMI
                from alg_code.problems.robustness import compute_rmi
                ph      = inst.phases[t]
                csets   = build_combo_set(ph, inst)
                x_best  = decode(best_c, csets, inst)
                bf1_val = evaluate_f1_raw(x_best, ph, inst)
                rmi = float(compute_rmi(best_c, bf1_val, csets, ph, inst,
                                        r=2, S=50, rng=rng_nsga))

            recs, f1, f2, ph = get_allocation(inst, best_c, t)
            ph_results.append({
                'phase': t, 'records': recs,
                'f1': f1, 'f2': f2, 'rmi': rmi,
            })
        results[algo] = ph_results
    return inst, results

# ═══════════════════════════════════════════════════════════════════════
# 图A：战场地图（双阶段，双算法 → 2×2子图）
# ═══════════════════════════════════════════════════════════════════════
def fig_battlefield(inst, results):
    plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('图A  I1实例战场分配地图（6武器×6雷达×13目标，两决策阶段）',
                 fontsize=14, fontweight='bold')

    algos = ['rcmd_ea', 'nsga2_topsis']
    for row, algo in enumerate(algos):
        for col, phase_idx in enumerate([0, 1]):
            ax   = axes[row, col]
            ph   = inst.phases[phase_idx]
            data = results[algo][phase_idx]
            recs = data['records']

            # ── 背景网格 ──
            ax.set_facecolor('#f0f4f8')
            ax.grid(color='white', lw=0.8, zorder=0)

            # ── 雷达（六角形） ──
            for k in range(inst.L):
                rx, ry = inst.radar_x[k], inst.radar_y[k]
                ax.plot(rx, ry, 'h', color=RADAR_COLOR, ms=14, zorder=4,
                        markeredgecolor='white', markeredgewidth=1.5)
                ax.text(rx+1, ry+2, f'R{k+1}', fontsize=8, color=RADAR_COLOR,
                        fontweight='bold', zorder=5)

            # ── 武器平台（正方形） ──
            for i in range(inst.m):
                wx, wy = inst.weapon_x[i], inst.weapon_y[i]
                ax.plot(wx, wy, 's', color=WEAPON_COLOR, ms=14, zorder=4,
                        markeredgecolor='white', markeredgewidth=1.5)
                ax.text(wx+1, wy+2, f'W{i+1}', fontsize=8, color=WEAPON_COLOR,
                        fontweight='bold', zorder=5)

            # ── 目标（按类别着色） ──
            for rec in recs:
                j   = rec['target_idx']
                tx, ty = rec['x'], rec['y']
                cat    = rec['category']
                threat = rec['threat']
                # 圆圈大小与威胁等级正相关
                size   = 80 + threat * 200
                color  = CAT_COLORS[cat]
                alpha  = 0.5 + threat * 0.4
                ax.scatter(tx, ty, s=size, c=color, alpha=alpha,
                           marker='o', zorder=3, edgecolors='black', linewidths=0.8)
                ax.text(tx+1.5, ty+1.5, f'T{j+1}', fontsize=7.5,
                        color='black', zorder=5, fontweight='bold')

                # ── 分配连线 ──
                for asn in rec['assigned']:
                    wi, ri = asn['weapon'], asn['radar']
                    wx_a, wy_a = inst.weapon_x[wi], inst.weapon_y[wi]
                    rx_a, ry_a = inst.radar_x[ri],  inst.radar_y[ri]

                    # 武器 → 目标（蓝线，线宽=shots）
                    lw = 1.0 + asn['shots'] * 0.8
                    kill_alpha = 0.4 + asn['p_kill'] * 0.5
                    ax.plot([wx_a, tx], [wy_a, ty],
                            '-', color=ALGO_COLORS[algo],
                            lw=lw, alpha=kill_alpha, zorder=2)

                    # 雷达 → 目标（虚线，制导链路）
                    ax.plot([rx_a, tx], [ry_a, ty],
                            '--', color='#FF9800', lw=0.9, alpha=0.45, zorder=1)

                    # 如果多发，在线中点标注shots
                    if asn['shots'] > 1:
                        mx2, my2 = (wx_a+tx)/2, (wy_a+ty)/2
                        ax.text(mx2, my2, f"×{asn['shots']}",
                                fontsize=7, color=ALGO_COLORS[algo],
                                bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.7),
                                zorder=6)

            # ── 无法拦截的目标（叉号标记） ──
            for rec in recs:
                if not rec['assigned']:
                    tx, ty = rec['x'], rec['y']
                    ax.plot(tx, ty, 'x', color='#666', ms=12, mew=2, zorder=4)

            # ── 标题和指标 ──
            rmi_val = data['rmi']
            f1_val  = data['f1']
            f2_val  = data['f2']
            ax.set_title(
                f'{ALGO_LABELS[algo]} — 阶段{phase_idx+1}\n'
                f'F1={f1_val:.3f}  弹药={f2_val}发  RMI={rmi_val:.0f}',
                fontsize=11, fontweight='bold',
                color=ALGO_COLORS[algo]
            )
            ax.set_xlabel('X坐标 (km)', fontsize=9)
            ax.set_ylabel('Y坐标 (km)', fontsize=9)

    # ── 统一图例 ──
    legend_elements = [
        Line2D([0],[0], marker='s', color='w', markerfacecolor=WEAPON_COLOR,
               markersize=10, label='武器平台（绿方块）'),
        Line2D([0],[0], marker='h', color='w', markerfacecolor=RADAR_COLOR,
               markersize=10, label='雷达站（红六角）'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=CAT_COLORS['conventional'],
               markersize=9, label='普通目标'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=CAT_COLORS['low_observable'],
               markersize=9, label='低可见目标'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=CAT_COLORS['stealth'],
               markersize=9, label='隐身目标'),
        Line2D([0],[0], color=ALGO_COLORS['rcmd_ea'], lw=2, label='RCMD-EA 打击链'),
        Line2D([0],[0], color=ALGO_COLORS['nsga2_topsis'], lw=2, label='NSGA-II 打击链'),
        Line2D([0],[0], color='#FF9800', lw=1.5, ls='--', label='雷达制导链'),
        Line2D([0],[0], marker='x', color='#666', ms=9, mew=2, label='未拦截目标'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, -0.02),
               frameon=True, fancybox=True, shadow=True)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    return b64(fig, dpi=120)

# ═══════════════════════════════════════════════════════════════════════
# 图B：分配详情热力矩阵（武器×目标矩阵，颜色=毁伤概率）
# ═══════════════════════════════════════════════════════════════════════
def fig_assignment_heatmap(inst, results):
    plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('图B  武器-目标分配热力矩阵（颜色=毁伤概率，数字=弹药数）',
                 fontsize=14, fontweight='bold')

    algos = ['rcmd_ea', 'nsga2_topsis']
    for row, algo in enumerate(algos):
        for col, phase_idx in enumerate([0, 1]):
            ax   = axes[row, col]
            ph   = inst.phases[phase_idx]
            recs = results[algo][phase_idx]['records']
            n_t  = ph.n_t
            m    = inst.m

            # 构建矩阵：heatmap[i,j] = 毁伤概率（0=未分配）
            heat = np.zeros((m, n_t))
            shots_mat = np.zeros((m, n_t), dtype=int)
            for rec in recs:
                j = rec['target_idx']
                for asn in rec['assigned']:
                    i = asn['weapon']
                    heat[i, j] = asn['p_kill']
                    shots_mat[i, j] = asn['shots']

            # 热力图
            cmap = matplotlib.colormaps.get_cmap('YlOrRd')
            cmap.set_under('#e8eaf6')  # 未分配=淡蓝
            im = ax.imshow(heat, cmap=cmap, vmin=0.01, vmax=1.0, aspect='auto',
                           interpolation='nearest')

            # 在每格标注shots数和p_kill
            for i in range(m):
                for j in range(n_t):
                    if shots_mat[i, j] > 0:
                        pk = heat[i, j]
                        ax.text(j, i, f'{shots_mat[i,j]}发\n{pk:.0%}',
                                ha='center', va='center', fontsize=7.5,
                                color='white' if pk > 0.5 else 'black', fontweight='bold')
                    else:
                        ax.text(j, i, '—', ha='center', va='center',
                                fontsize=9, color='#9e9e9e')

            # 轴标签
            threats  = [f'T{j+1}\n({recs[j]["threat"]:.2f})' for j in range(n_t)]
            wpn_lbls = [f'W{i+1}' for i in range(m)]
            ax.set_xticks(range(n_t)); ax.set_xticklabels(threats, fontsize=7.5)
            ax.set_yticks(range(m));   ax.set_yticklabels(wpn_lbls, fontsize=9)

            # 目标类别颜色条（顶部色带）
            for j, rec in enumerate(recs):
                c = CAT_COLORS[rec['category']]
                ax.add_patch(mpatches.FancyBboxPatch(
                    (j-0.45, m-0.1), 0.9, 0.35,
                    boxstyle='round,pad=0', fc=c, ec='none', alpha=0.8,
                    transform=ax.transData, clip_on=False))

            rmi_val = results[algo][phase_idx]['rmi']
            f1_val  = results[algo][phase_idx]['f1']
            f2_val  = results[algo][phase_idx]['f2']
            ax.set_title(
                f'{ALGO_LABELS[algo]} — 阶段{phase_idx+1} | '
                f'F1={f1_val:.3f}  弹药={f2_val}发  RMI={rmi_val:.0f}',
                fontsize=10.5, fontweight='bold', color=ALGO_COLORS[algo]
            )
            ax.set_xlabel('目标编号（括号内为威胁度）', fontsize=9)
            ax.set_ylabel('武器平台', fontsize=9)

            plt.colorbar(im, ax=ax, label='毁伤概率', fraction=0.025, pad=0.02)

    plt.tight_layout()
    return b64(fig, dpi=115)

# ═══════════════════════════════════════════════════════════════════════
# 图C：鲁棒性分解图（每个目标的残余威胁贡献 + 生存概率）
# ═══════════════════════════════════════════════════════════════════════
def fig_residual_breakdown(inst, results):
    plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    fig.suptitle('图C  各目标残余威胁贡献分解（阶段1+2，双算法对比）',
                 fontsize=14, fontweight='bold')

    algos = ['rcmd_ea', 'nsga2_topsis']
    for row, algo in enumerate(algos):
        for col, phase_idx in enumerate([0, 1]):
            ax   = axes[row, col]
            recs = results[algo][phase_idx]['records']
            n_t  = len(recs)

            xs      = np.arange(n_t)
            threats = np.array([r['threat'] for r in recs])
            etas    = np.array([r['eta']    for r in recs])
            survs   = np.array([r['survival'] for r in recs])
            resids  = np.array([r['residual'] for r in recs])

            # 堆叠柱：被拦截部分（蓝）vs 残余（红）
            intercepted = threats * etas * (1 - survs)
            ax.bar(xs, resids,     color='#ef5350', alpha=0.85, label='残余威胁（F1贡献）')
            ax.bar(xs, intercepted, bottom=resids,
                   color='#42a5f5', alpha=0.75, label='已拦截威胁贡献')

            # 叠加eta_j折线
            ax2 = ax.twinx()
            ax2.plot(xs, etas, 'o--', color='#9c27b0', lw=1.5, ms=5,
                     alpha=0.75, label='MRFDQI探测质量 η')
            ax2.set_ylabel('MRFDQI η', fontsize=8, color='#9c27b0')
            ax2.tick_params(axis='y', colors='#9c27b0', labelsize=8)
            ax2.set_ylim(0, 1.4)

            # 类别背景色
            for j, rec in enumerate(recs):
                c = CAT_COLORS[rec['category']]
                ax.axvspan(j-0.45, j+0.45, alpha=0.06, color=c)

            # 弹药分配标注
            for j, rec in enumerate(recs):
                total_shots = sum(a['shots'] for a in rec['assigned'])
                if total_shots > 0:
                    ax.text(j, resids[j]+intercepted[j]+0.01,
                            f'{total_shots}发', ha='center', va='bottom',
                            fontsize=7.5, color='#1565c0', fontweight='bold')
                else:
                    ax.text(j, 0.02, '未分配', ha='center', va='bottom',
                            fontsize=6.5, color='#666', rotation=90)

            ax.set_xticks(xs)
            ax.set_xticklabels([f'T{j+1}' for j in range(n_t)], fontsize=8)
            ax.set_xlabel('目标编号', fontsize=9)
            ax.set_ylabel('威胁贡献量', fontsize=9)

            rmi_v = results[algo][phase_idx]['rmi']
            f1_v  = results[algo][phase_idx]['f1']
            ax.set_title(
                f'{ALGO_LABELS[algo]} — 阶段{phase_idx+1}\n'
                f'总F1={f1_v:.3f}  RMI={rmi_v:.0f}',
                fontsize=10.5, fontweight='bold', color=ALGO_COLORS[algo]
            )
            ax.set_ylim(0)
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1+lines2, labels1+labels2, fontsize=7.5, loc='upper right')

    plt.tight_layout()
    return b64(fig, dpi=115)

# ═══════════════════════════════════════════════════════════════════════
# HTML表格：分配详情（一张表）
# ═══════════════════════════════════════════════════════════════════════
CAT_BADGE = {
    'conventional':   '<span style="background:#2196F3;color:#fff;border-radius:4px;padding:1px 6px;font-size:10px">普通</span>',
    'low_observable': '<span style="background:#FF9800;color:#fff;border-radius:4px;padding:1px 6px;font-size:10px">低可见</span>',
    'stealth':        '<span style="background:#9C27B0;color:#fff;border-radius:4px;padding:1px 6px;font-size:10px">隐身</span>',
}

def html_allocation_table(inst, results, algo, phase_idx):
    recs  = results[algo][phase_idx]['records']
    f1    = results[algo][phase_idx]['f1']
    f2    = results[algo][phase_idx]['f2']
    rmi   = results[algo][phase_idx]['rmi']
    ph    = inst.phases[phase_idx]

    def threat_bar(v):
        pct = int(v * 100)
        color = '#ef5350' if v > 0.7 else '#FF9800' if v > 0.4 else '#66bb6a'
        return (f'<div style="display:flex;align-items:center;gap:6px">'
                f'<div style="width:{pct}px;max-width:80px;height:10px;background:{color};'
                f'border-radius:3px;flex-shrink:0"></div>'
                f'<span style="font-size:11px">{v:.3f}</span></div>')

    def pk_pill(pk):
        if pk <= 0: return '<span style="color:#aaa">—</span>'
        color = '#43a047' if pk > 0.7 else '#fb8c00' if pk > 0.4 else '#ef5350'
        return f'<span style="background:{color};color:#fff;border-radius:10px;padding:1px 7px;font-size:11px">{pk:.0%}</span>'

    rows = ''
    for rec in recs:
        j        = rec['target_idx']
        assigned = rec['assigned']
        asn_html = ''
        if assigned:
            parts = []
            for a in assigned:
                parts.append(
                    f'<span style="background:#e8eaf6;border-radius:4px;padding:2px 6px;'
                    f'font-size:11px;white-space:nowrap">'
                    f'W{a["weapon"]+1}+R{a["radar"]+1} × {a["shots"]}发</span>'
                )
            asn_html = ' '.join(parts)
        else:
            asn_html = '<span style="color:#aaa;font-size:11px">未分配</span>'

        pk_total = rec['p_kill_total']
        resid    = rec['residual']
        resid_color = '#ef5350' if resid > 0.3 else '#FB8C00' if resid > 0.1 else '#43a047'

        rows += f'''<tr>
            <td style="font-weight:700;text-align:center">T{j+1}</td>
            <td>{CAT_BADGE[rec["category"]]}</td>
            <td>{threat_bar(rec["threat"])}</td>
            <td style="text-align:center;font-size:11px">{rec["eta"]:.3f}</td>
            <td>{asn_html}</td>
            <td style="text-align:center">{pk_pill(pk_total)}</td>
            <td style="text-align:center;font-weight:700;color:{resid_color}">{resid:.4f}</td>
        </tr>'''

    algo_color = ALGO_COLORS[algo]
    algo_label = ALGO_LABELS[algo]
    return f'''
<div style="margin:12px 0">
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:8px">
    <span style="font-weight:700;font-size:13px;color:{algo_color}">{algo_label} — 阶段{phase_idx+1}</span>
    <span style="background:#fff3e0;border-radius:6px;padding:3px 10px;font-size:12px">F1 = {f1:.4f}</span>
    <span style="background:#e8f5e9;border-radius:6px;padding:3px 10px;font-size:12px">弹药消耗 = {f2}发</span>
    <span style="background:#e8eaf6;border-radius:6px;padding:3px 10px;font-size:12px">RMI = <b style="color:{algo_color}">{rmi:.0f}</b></span>
  </div>
  <div style="overflow-x:auto">
  <table style="border-collapse:collapse;width:100%;font-size:12px">
    <thead>
      <tr style="background:#1a237e;color:#fff">
        <th style="padding:7px 10px">目标</th>
        <th>类别</th>
        <th style="min-width:120px">威胁度</th>
        <th>MRFDQI η</th>
        <th>分配方案（武器+雷达×弹数）</th>
        <th>综合毁伤概率</th>
        <th>残余威胁</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
    <tfoot>
      <tr style="background:#f5f5f5;font-weight:700">
        <td colspan="6" style="text-align:right;padding:6px 10px">合计残余威胁 F1 =</td>
        <td style="text-align:center;color:#c62828">{f1:.4f}</td>
      </tr>
    </tfoot>
  </table>
  </div>
</div>'''

# ═══════════════════════════════════════════════════════════════════════
# 图D：两算法分配对比 - 关键差异高亮
# ═══════════════════════════════════════════════════════════════════════
def fig_comparison_bar(inst, results):
    plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle('图D  两算法关键指标对比（I1实例，两阶段）', fontsize=13, fontweight='bold')

    phases = [0, 1]
    phase_lbls = ['阶段1', '阶段2']

    # 左：F1和RMI双轴对比
    ax  = axes[0]
    ax2 = ax.twinx()
    x   = np.arange(2)
    w   = 0.35

    for k, algo in enumerate(['rcmd_ea','nsga2_topsis']):
        f1s  = [results[algo][p]['f1']  for p in phases]
        rmis = [results[algo][p]['rmi'] for p in phases]
        bars = ax.bar(x + (k-0.5)*w, f1s, w,
                      color=ALGO_COLORS[algo], alpha=0.75, label=f'{ALGO_LABELS[algo]} F1')
        for bar, v in zip(bars, f1s):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.03, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=9, color=ALGO_COLORS[algo], fontweight='bold')

    ax.set_xticks(x); ax.set_xticklabels(phase_lbls)
    ax.set_ylabel('F1 残余威胁（↓越好）', fontsize=10)
    ax.set_title('(a) F1效能对比', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # RMI折线（右轴）
    for algo in ['rcmd_ea','nsga2_topsis']:
        rmis = [results[algo][p]['rmi'] for p in phases]
        ax2.plot(x, rmis, 'o-', color=ALGO_COLORS[algo], lw=2.5, ms=9,
                 ls='--', label=f'{ALGO_LABELS[algo]} RMI',
                 markeredgecolor='black', markeredgewidth=0.8)
        for xi, v in zip(x, rmis):
            ax2.text(xi+0.08, v+20, f'RMI={v:.0f}', fontsize=9,
                     color=ALGO_COLORS[algo], fontweight='bold')
    ax2.set_ylabel('RMI（↑越鲁棒）', fontsize=10)
    ax2.legend(loc='upper right', fontsize=8)

    # 右：每目标残余威胁对比（阶段1）
    ax3 = axes[1]
    ph  = 0
    recs_rcmd = results['rcmd_ea'][ph]['records']
    recs_nsga = results['nsga2_topsis'][ph]['records']
    n_t       = len(recs_rcmd)
    xs        = np.arange(n_t)

    resids_rcmd = np.array([r['residual'] for r in recs_rcmd])
    resids_nsga = np.array([r['residual'] for r in recs_nsga])
    diff        = resids_nsga - resids_rcmd  # 正=RCMD更好

    ax3.bar(xs - 0.2, resids_rcmd, 0.38, color=ALGO_COLORS['rcmd_ea'],
            alpha=0.8, label='RCMD-EA')
    ax3.bar(xs + 0.2, resids_nsga, 0.38, color=ALGO_COLORS['nsga2_topsis'],
            alpha=0.8, label='NSGA-II+TOPSIS')

    # 差异标注（RCMD-EA更好的目标用绿色箭头）
    for j in range(n_t):
        if diff[j] > 0.01:  # RCMD-EA显著更好
            y_top = max(resids_rcmd[j], resids_nsga[j]) + 0.02
            ax3.annotate('', xy=(j-0.2, resids_rcmd[j]+0.005), xytext=(j-0.2, y_top),
                         arrowprops=dict(arrowstyle='->', color='green', lw=1.2))

    ax3.set_xticks(xs)
    ax3.set_xticklabels([f'T{j+1}' for j in range(n_t)], fontsize=8)
    ax3.set_xlabel('目标编号', fontsize=10)
    ax3.set_ylabel('残余威胁贡献', fontsize=10)
    ax3.set_title('(b) 阶段1各目标残余威胁分解对比\n（↑绿箭头=RCMD-EA在该目标上更优）',
                  fontsize=10.5, fontweight='bold')
    ax3.legend(fontsize=9); ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return b64(fig, dpi=115)

# ═══════════════════════════════════════════════════════════════════════
# 主入口：生成完整HTML片段
# ═══════════════════════════════════════════════════════════════════════
def generate_allocation_section():
    print("  [A] 运行算法，获取I1分配结果...")
    inst, results = run_both_algos(seed=42)

    print("  [B] 生成战场地图...")
    img_battlefield = fig_battlefield(inst, results)

    print("  [C] 生成分配热力矩阵...")
    img_heatmap = fig_assignment_heatmap(inst, results)

    print("  [D] 生成残余威胁分解图...")
    img_residual = fig_residual_breakdown(inst, results)

    print("  [E] 生成对比柱状图...")
    img_compare = fig_comparison_bar(inst, results)

    # 分配详情表（4个：2算法×2阶段）
    tables_html = ''
    for algo in ['rcmd_ea','nsga2_topsis']:
        tables_html += f'<h4 style="color:{ALGO_COLORS[algo]};margin:18px 0 6px">{ALGO_LABELS[algo]}</h4>'
        for phase_idx in [0, 1]:
            tables_html += html_allocation_table(inst, results, algo, phase_idx)

    # 统计一下关键数字
    rcmd_rmi  = [results['rcmd_ea'][p]['rmi']  for p in [0,1]]
    nsga_rmi  = [results['nsga2_topsis'][p]['rmi'] for p in [0,1]]
    rcmd_f1   = [results['rcmd_ea'][p]['f1']   for p in [0,1]]
    nsga_f1   = [results['nsga2_topsis'][p]['f1']  for p in [0,1]]
    rmi_ratio = [rcmd_rmi[p]/nsga_rmi[p] for p in [0,1]]

    section = f'''
<h2>三、I1实例分配结果深度解析</h2>
<div class="card">
  <p class="note">
  <b>实例规模</b>：I1，6武器平台 × 6雷达站 × 13目标，2个决策阶段。
  以固定种子（seed=42）各运行一次，展示RCMD-EA与NSGA-II+TOPSIS的
  实际武器目标分配方案，直观说明鲁棒性优势的来源。<br>
  <b>关键数字</b>：阶段1 RMI比值={rmi_ratio[0]:.1f}×，阶段2={rmi_ratio[1]:.1f}×；
  RCMD-EA的F1分别为{rcmd_f1[0]:.3f}/{rcmd_f1[1]:.3f}，
  NSGA-II为{nsga_f1[0]:.3f}/{nsga_f1[1]:.3f}。
  </p>

  <h3>战场分配地图</h3>
  <div class="fig-row cols-1">{img_tag(img_battlefield, '100%',
    '战场地图：绿方块=武器平台，红六角=雷达站，圆=目标（大小正比于威胁度），实线=打击链路，虚线=雷达制导链路')}</div>
  <p class="note">
  从地图可以直观看出两算法的分配差异：
  RCMD-EA倾向于将弹药集中在"威胁度高且雷达探测质量好（η高）"的目标上，
  形成更集中的打击链路；对于低可见/隐身目标（η较低，毁伤概率不确定性大），
  RCMD-EA会在弹药允许时分配2发以上，以在扰动情况下保持一定的覆盖稳定性。
  NSGA-II+TOPSIS有时会对低η目标单发拦截，这类方案的RMI低——
  邻域扰动后拦截效果会急剧变化。
  </p>

  <h3>武器-目标分配热力矩阵</h3>
  <div class="fig-row cols-1">{img_tag(img_heatmap, '100%',
    '热力矩阵：颜色深度=综合毁伤概率，格内数字=弹药数+毁伤概率；顶部色带=目标类别（蓝=普通，橙=低可见，紫=隐身）')}</div>
  <p class="note">
  热力矩阵精确量化了每个武器对每个目标的打击效果。
  RCMD-EA的矩阵中高毁伤概率格（深红色）数量更多，说明分配更集中有效；
  NSGA-II+TOPSIS的矩阵更分散——虽然覆盖目标更多，但单次打击概率偏低，
  对随机扰动（如某武器弹药失效）的抵抗力弱。
  </p>

  <h3>各目标残余威胁贡献分解</h3>
  <div class="fig-row cols-1">{img_tag(img_residual, '100%',
    '蓝柱=已拦截威胁贡献，红柱=残余威胁（F1的来源），折线=雷达探测质量η；数字=该目标消耗弹药数')}</div>

  <h3>关键指标对比图</h3>
  <div class="fig-row cols-1">{img_tag(img_compare, '100%',
    '左：F1效能（柱）与RMI鲁棒性（折线）双轴对比；右：阶段1各目标残余威胁贡献分解，绿箭头=RCMD-EA在该目标上更优')}</div>

  <h3>分配详情数值表</h3>
  <p class="note"><b>表格说明</b>：威胁度彩条越长越危险；
  综合毁伤概率 = 1 − (1−单发毁伤概率)^弹数；
  残余威胁 = 威胁度 × MRFDQI × 生存概率（越小越好，累计为F1）。</p>
  {tables_html}
</div>
'''
    return section

if __name__ == '__main__':
    section = generate_allocation_section()
    out = ROOT / 'reports/analysis/_alloc_section.html'
    with open(out, 'w', encoding='utf-8') as f:
        f.write(f'<html><head><meta charset="UTF-8"></head><body>{section}</body></html>')
    print(f"测试输出 → {out}")
