from __future__ import annotations
import argparse
import copy
import csv
import datetime
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
try:
    import final_unified_model as m
except Exception:
    import q1 as m

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def make_dated_outdir(base: str) -> str:
    today = datetime.date.today().strftime('%Y%m%d')
    candidate = f'{base}_{today}'
    if not os.path.exists(candidate):
        return candidate
    k = 2
    while True:
        cand2 = f'{candidate}_{k}'
        if not os.path.exists(cand2):
            return cand2
        k += 1

def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    ensure_dir(os.path.dirname(path) or '.')
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def mean(xs: Sequence[float]) -> float:
    xs2 = [x for x in xs if math.isfinite(x)]
    return sum(xs2) / len(xs2) if xs2 else float('nan')

def stdev(xs: Sequence[float]) -> float:
    xs2 = [x for x in xs if math.isfinite(x)]
    n = len(xs2)
    if n <= 1:
        return float('nan')
    mu = sum(xs2) / n
    v = sum(((x - mu) ** 2 for x in xs2)) / (n - 1)
    return math.sqrt(max(0.0, v))

def percentile(xs: Sequence[float], q: float) -> float:
    xs2 = sorted([x for x in xs if math.isfinite(x)])
    if not xs2:
        return float('nan')
    q = clamp(q, 0.0, 1.0)
    if len(xs2) == 1:
        return xs2[0]
    pos = q * (len(xs2) - 1)
    i = int(math.floor(pos))
    j = int(math.ceil(pos))
    if i == j:
        return xs2[i]
    w = pos - i
    return xs2[i] * (1.0 - w) + xs2[j] * w

def rankdata(xs: Sequence[float]) -> List[float]:
    n = len(xs)
    order = sorted(range(n), key=lambda i: xs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks

def pearsonr(x: Sequence[float], y: Sequence[float]) -> float:
    n = min(len(x), len(y))
    if n <= 1:
        return float('nan')
    xs = [float(xi) for xi in x[:n]]
    ys = [float(yi) for yi in y[:n]]
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = math.sqrt(sum(((xi - mx) ** 2 for xi in xs)))
    sy = math.sqrt(sum(((yi - my) ** 2 for yi in ys)))
    if sx <= 1e-12 or sy <= 1e-12:
        return float('nan')
    cov = sum(((xi - mx) * (yi - my) for xi, yi in zip(xs, ys)))
    return cov / (sx * sy)

def spearmanr(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) <= 1:
        return float('nan')
    rx = rankdata(x)
    ry = rankdata(y)
    return pearsonr(rx, ry)

def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def mann_whitney_u_pvalue(x: Sequence[float], y: Sequence[float]) -> float:
    x2 = [float(v) for v in x if math.isfinite(v)]
    y2 = [float(v) for v in y if math.isfinite(v)]
    n1, n2 = (len(x2), len(y2))
    if n1 == 0 or n2 == 0:
        return float('nan')
    pooled = x2 + y2
    r = rankdata(pooled)
    r1 = sum(r[:n1])
    u1 = n1 * n2 + n1 * (n1 + 1) / 2.0 - r1
    mu = n1 * n2 / 2.0
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if sigma <= 1e-12:
        return float('nan')
    z = (u1 - mu) / sigma
    p = 2.0 * min(normal_cdf(z), 1.0 - normal_cdf(z))
    return clamp(p, 0.0, 1.0)

def paired_bootstrap_test(a: Sequence[float], b: Sequence[float], *, n_boot: int, seed: int) -> Tuple[float, float, float]:
    rnd = random.Random(int(seed))
    pairs = [(float(x), float(y)) for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)]
    n = len(pairs)
    if n == 0:
        return (float('nan'), float('nan'), float('nan'))
    diffs = [y - x for x, y in pairs]
    delta = sum(diffs) / n
    boots: List[float] = []
    for _ in range(max(1, int(n_boot))):
        s = 0.0
        for _k in range(n):
            x, y = pairs[rnd.randrange(n)]
            s += y - x
        boots.append(s / n)
    lo = percentile(boots, 0.025)
    hi = percentile(boots, 0.975)
    return (delta, lo, hi)

def paired_bootstrap_pvalue(a: Sequence[float], b: Sequence[float], *, n_boot: int, seed: int) -> float:
    rnd = random.Random(int(seed))
    pairs = [(float(x), float(y)) for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)]
    n = len(pairs)
    if n == 0:
        return float('nan')
    boots: List[float] = []
    for _ in range(max(1, int(n_boot))):
        s = 0.0
        for _k in range(n):
            x, y = pairs[rnd.randrange(n)]
            s += y - x
        boots.append(s / n)
    p_le = sum((1 for d in boots if d <= 0.0)) / len(boots)
    p_ge = sum((1 for d in boots if d >= 0.0)) / len(boots)
    return clamp(2.0 * min(p_le, p_ge), 0.0, 1.0)

def make_constant_step(*, dt_s: float, foreground: str, screen_on: bool, brightness: float, refresh_hz: float, net_requests_per_s: float, signal_quality: float, network_type: str, gps_on: bool, bt_scan: bool, background_wake_per_s: float, ambient_temp_C: Optional[float], oled_content_factor: Optional[float]=None) -> m.ScenarioStep:
    return m.ScenarioStep(dt_s=dt_s, screen_on=screen_on, brightness=brightness, refresh_hz=refresh_hz, oled_content_factor=oled_content_factor, foreground=foreground, net_requests_per_s=net_requests_per_s, signal_quality=signal_quality, network_type=network_type, gps_on=gps_on, bt_scan=bt_scan, background_wake_per_s=background_wake_per_s, ambient_temp_C=ambient_temp_C)

def repeat_steps(step: m.ScenarioStep, hours: float) -> List[m.ScenarioStep]:
    n = max(1, int(hours * 3600.0 / max(1e-06, float(step.dt_s))))
    return [step for _ in range(n)]

def scenario_suite(dt_s: float, suite: str='core') -> List[Tuple[str, m.ScenarioStep, bool]]:
    core: List[Tuple[str, m.ScenarioStep, bool]] = [('重度游戏_室温', make_constant_step(dt_s=dt_s, foreground='game', screen_on=True, brightness=0.85, refresh_hz=120.0, net_requests_per_s=0.1, signal_quality=0.85, network_type='wifi', gps_on=False, bt_scan=False, background_wake_per_s=0.1, ambient_temp_C=22.0, oled_content_factor=0.9), True), ('短视频_室温', make_constant_step(dt_s=dt_s, foreground='video', screen_on=True, brightness=0.65, refresh_hz=120.0, net_requests_per_s=0.7, signal_quality=0.9, network_type='wifi', gps_on=False, bt_scan=False, background_wake_per_s=0.12, ambient_temp_C=22.0, oled_content_factor=0.8), True), ('导航_弱信号_夏季', make_constant_step(dt_s=dt_s, foreground='navigation', screen_on=True, brightness=0.85, refresh_hz=90.0, net_requests_per_s=0.25, signal_quality=0.3, network_type='cell', gps_on=True, bt_scan=True, background_wake_per_s=0.18, ambient_temp_C=35.0, oled_content_factor=0.9), True), ('待机推送_冬季', make_constant_step(dt_s=dt_s, foreground='idle', screen_on=False, brightness=0.0, refresh_hz=60.0, net_requests_per_s=0.02, signal_quality=0.6, network_type='cell', gps_on=False, bt_scan=False, background_wake_per_s=0.1, ambient_temp_C=-5.0, oled_content_factor=None), True), ('室内浏览_高亮度', make_constant_step(dt_s=dt_s, foreground='social', screen_on=True, brightness=0.95, refresh_hz=120.0, net_requests_per_s=0.3, signal_quality=0.9, network_type='wifi', gps_on=False, bt_scan=False, background_wake_per_s=0.2, ambient_temp_C=22.0, oled_content_factor=1.0), True)]
    if suite == 'core':
        return core
    if suite != 'extended':
        raise ValueError(f'未知 suite: {suite}')
    ext = list(core)
    ext += [('长视频_暗色主题_WIFI', make_constant_step(dt_s=dt_s, foreground='video', screen_on=True, brightness=0.35, refresh_hz=60.0, net_requests_per_s=0.08, signal_quality=0.95, network_type='wifi', gps_on=False, bt_scan=False, background_wake_per_s=0.08, ambient_temp_C=22.0, oled_content_factor=0.7), True), ('电子书阅读_低亮度', make_constant_step(dt_s=dt_s, foreground='idle', screen_on=True, brightness=0.18, refresh_hz=60.0, net_requests_per_s=0.01, signal_quality=0.95, network_type='wifi', gps_on=False, bt_scan=False, background_wake_per_s=0.05, ambient_temp_C=22.0, oled_content_factor=0.75), True), ('后台下载_锁屏_WIFI', make_constant_step(dt_s=dt_s, foreground='idle', screen_on=False, brightness=0.0, refresh_hz=60.0, net_requests_per_s=0.55, signal_quality=0.95, network_type='wifi', gps_on=False, bt_scan=False, background_wake_per_s=0.16, ambient_temp_C=22.0, oled_content_factor=None), True)]
    return ext

@dataclass
class ScenarioMetrics:
    t_empty_h: float
    t_end_h: float
    soc_end: float
    T_max_C: float
    perf_min: float
    avg_p_sys_W: float
    terminated_reason: str

def dt_series_from_rows(rows: List[Dict[str, float]]) -> List[float]:
    if len(rows) <= 1:
        return [0.0]
    dts: List[float] = []
    for i in range(len(rows) - 1):
        dts.append(max(0.0, float(rows[i + 1]['t_s']) - float(rows[i]['t_s'])))
    dts_sorted = sorted(dts)
    dt_last = dts_sorted[len(dts_sorted) // 2] if dts_sorted else 0.0
    dts.append(dt_last)
    return dts

def integrate_avg_p(rows: List[Dict[str, float]]) -> float:
    if not rows:
        return float('nan')
    dts = dt_series_from_rows(rows)
    e_J = 0.0
    t_s = 0.0
    for row, dt in zip(rows, dts):
        t_s += dt
        e_J += float(row.get('p_sys_W', 0.0)) * dt
    return e_J / t_s if t_s > 1e-09 else float('nan')

def run_metrics(params: m.ModelParams, step: m.ScenarioStep, *, max_hours: float, extend_until_empty: bool=True) -> ScenarioMetrics:
    sim = m.simulate(params, repeat_steps(step, hours=1.0), z0=1.0, extend_with_idle=extend_until_empty, idle_step=step, max_time_s=float(max_hours) * 3600.0)
    rows: List[Dict[str, float]] = sim['rows']
    t_empty_s = sim.get('t_empty_s')
    terminated_reason = str(sim.get('terminated_reason', 'unknown'))
    t_end_s = float(rows[-1]['t_s']) if rows else 0.0
    soc_end = float(rows[-1]['soc']) if rows else float('nan')
    T_max = max((float(r.get('T_C', 0.0)) for r in rows), default=float('nan'))
    perf_min = min((float(r.get('perf_factor', 1.0)) for r in rows), default=float('nan'))
    avg_p = integrate_avg_p(rows)
    t_empty_h = float('nan') if t_empty_s is None else float(t_empty_s) / 3600.0
    return ScenarioMetrics(t_empty_h=t_empty_h, t_end_h=t_end_s / 3600.0, soc_end=soc_end, T_max_C=T_max, perf_min=perf_min, avg_p_sys_W=float(avg_p), terminated_reason=terminated_reason)

def get_by_path(obj: object, path: str) -> object:
    cur = obj
    for part in path.split('.'):
        cur = getattr(cur, part)
    return cur

def set_by_path(obj: object, path: str, value: object) -> None:
    parts = path.split('.')
    cur = obj
    for part in parts[:-1]:
        cur = getattr(cur, part)
    setattr(cur, parts[-1], value)

@dataclass
class ParamSpec:
    name: str
    kind: str
    path: str
    dist: str
    rel_sigma: float = 0.0
    abs_sigma: float = 0.0
    abs_half_range: float = 0.0
    choices: Tuple[float, ...] = ()
    clamp_lo: Optional[float] = None
    clamp_hi: Optional[float] = None

def sample_param(rnd: random.Random, nominal: float, spec: ParamSpec) -> float:
    if spec.dist == 'lognormal_rel':
        s = max(0.0, float(spec.rel_sigma))
        if s <= 1e-12:
            v = nominal
        else:
            sigma_ln = s
            v = nominal * math.exp(rnd.gauss(0.0, sigma_ln))
    elif spec.dist == 'normal_abs':
        v = float(nominal) + rnd.gauss(0.0, float(spec.abs_sigma))
    elif spec.dist == 'uniform_abs':
        r = float(spec.abs_half_range)
        v = float(nominal) + rnd.uniform(-r, r)
    elif spec.dist == 'discrete':
        if not spec.choices:
            v = nominal
        else:
            v = float(rnd.choice(list(spec.choices)))
    else:
        raise ValueError(f'未知分布: {spec.dist}')
    if spec.clamp_lo is not None:
        v = max(float(spec.clamp_lo), float(v))
    if spec.clamp_hi is not None:
        v = min(float(spec.clamp_hi), float(v))
    return float(v)

def compute_oat_sensitivity(baseline: float, new: float, delta_frac: float) -> float:
    if not (math.isfinite(baseline) and math.isfinite(new)):
        return float('nan')
    if baseline <= 1e-09:
        return float('nan')
    if abs(delta_frac) <= 1e-12:
        return float('nan')
    return (new - baseline) / baseline / delta_frac

def default_param_specs() -> List[ParamSpec]:
    return [ParamSpec(name='容量健康度', kind='model', path='battery.health_capacity_factor', dist='lognormal_rel', rel_sigma=0.05, clamp_lo=0.6, clamp_hi=1.05), ParamSpec(name='内阻健康度', kind='model', path='battery.health_resistance_factor', dist='lognormal_rel', rel_sigma=0.08, clamp_lo=0.8, clamp_hi=3.0), ParamSpec(name='DC-DC效率', kind='model', path='battery.eta_dc', dist='normal_abs', abs_sigma=0.015, clamp_lo=0.8, clamp_hi=0.98), ParamSpec(name='散热系数h', kind='model', path='thermal.h_W_per_C', dist='lognormal_rel', rel_sigma=0.1, clamp_lo=0.2, clamp_hi=3.0), ParamSpec(name='等效热容Cth', kind='model', path='thermal.C_th_J_per_C', dist='lognormal_rel', rel_sigma=0.12, clamp_lo=50.0, clamp_hi=400.0), ParamSpec(name='热限频起点', kind='model', path='thermal.T_throttle_start_C', dist='normal_abs', abs_sigma=1.5, clamp_lo=30.0, clamp_hi=50.0), ParamSpec(name='最低性能因子', kind='model', path='thermal.throttle_min_factor', dist='normal_abs', abs_sigma=0.05, clamp_lo=0.35, clamp_hi=0.85), ParamSpec(name='尾能量tau_tail', kind='model', path='radio.tau_tail_s', dist='lognormal_rel', rel_sigma=0.2, clamp_lo=2.0, clamp_hi=40.0), ParamSpec(name='无线ACTIVE功耗', kind='model', path='radio.p_active_W', dist='lognormal_rel', rel_sigma=0.12, clamp_lo=0.4, clamp_hi=3.0), ParamSpec(name='耦合:渲染', kind='model', path='interaction.k_screen_render_W', dist='lognormal_rel', rel_sigma=0.18, clamp_lo=0.0, clamp_hi=2.0), ParamSpec(name='耦合:弱信号×后台', kind='model', path='interaction.k_radio_bg_mult', dist='lognormal_rel', rel_sigma=0.18, clamp_lo=0.0, clamp_hi=2.0), ParamSpec(name='耦合:温度漏损', kind='model', path='interaction.k_thermal_leak_W_per_C', dist='lognormal_rel', rel_sigma=0.25, clamp_lo=0.0, clamp_hi=0.1), ParamSpec(name='调度激进度', kind='model', path='os.sched_aggressiveness', dist='normal_abs', abs_sigma=0.06, clamp_lo=0.7, clamp_hi=1.4), ParamSpec(name='外界温度', kind='step', path='ambient_temp_C', dist='uniform_abs', abs_half_range=5.0, clamp_lo=-20.0, clamp_hi=45.0), ParamSpec(name='屏幕亮度', kind='step', path='brightness', dist='uniform_abs', abs_half_range=0.1, clamp_lo=0.0, clamp_hi=1.0), ParamSpec(name='刷新率', kind='step', path='refresh_hz', dist='discrete', choices=(60.0, 90.0, 120.0), clamp_lo=30.0, clamp_hi=165.0), ParamSpec(name='网络请求率', kind='step', path='net_requests_per_s', dist='lognormal_rel', rel_sigma=0.3, clamp_lo=0.0, clamp_hi=5.0), ParamSpec(name='信号质量', kind='step', path='signal_quality', dist='uniform_abs', abs_half_range=0.15, clamp_lo=0.0, clamp_hi=1.0), ParamSpec(name='后台唤醒率', kind='step', path='background_wake_per_s', dist='lognormal_rel', rel_sigma=0.35, clamp_lo=0.0, clamp_hi=5.0), ParamSpec(name='OLED内容系数', kind='step', path='oled_content_factor', dist='uniform_abs', abs_half_range=0.1, clamp_lo=0.4, clamp_hi=1.2)]

def apply_assumption_variant(name: str, params: m.ModelParams, step: m.ScenarioStep) -> None:
    if name == 'baseline':
        return
    if name == 'no_interaction':
        params.interaction.k_screen_render_W = 0.0
        params.interaction.k_radio_bg_mult = 0.0
        params.interaction.k_thermal_leak_W_per_C = 0.0
        return
    if name == 'aggressive_throttle':
        params.thermal.T_throttle_start_C = float(params.thermal.T_throttle_start_C) - 3.0
        params.thermal.T_throttle_full_C = float(params.thermal.T_throttle_full_C) - 2.0
        params.thermal.throttle_min_factor = clamp(float(params.thermal.throttle_min_factor) - 0.1, 0.35, 0.9)
        return
    if name == 'longer_radio_tail':
        params.radio.tau_tail_s = clamp(float(params.radio.tau_tail_s) * 1.5, 2.0, 60.0)
        params.radio.tau_active_hold_s = clamp(float(params.radio.tau_active_hold_s) * 1.2, 0.5, 10.0)
        return
    if name == 'weaker_cooling':
        params.thermal.h_W_per_C = clamp(float(params.thermal.h_W_per_C) * 0.8, 0.2, 3.0)
        return
    raise ValueError(f'未知假设变体: {name}')

def available_variants() -> List[str]:
    return ['baseline', 'no_interaction', 'aggressive_throttle', 'longer_radio_tail']

def configure_cn_font() -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except Exception:
        return
    candidates = ['Microsoft YaHei', 'Microsoft YaHei UI', 'SimHei', 'SimSun', 'DengXian', 'PingFang SC', 'Noto Sans CJK SC', 'Source Han Sans SC']
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((name for name in candidates if name in available), None)
    if chosen is not None:
        plt.rcParams['font.sans-serif'] = [chosen] + list(plt.rcParams.get('font.sans-serif', []))
    plt.rcParams['axes.unicode_minus'] = False

def plot_barh(out_png: str, title: str, labels: List[str], values: List[float], xlabel: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    configure_cn_font()
    items = list(zip(labels, values))
    items.sort(key=lambda x: abs(float(x[1])), reverse=True)
    items = items[:16]
    labels2 = [x[0] for x in items][::-1]
    values2 = [float(x[1]) for x in items][::-1]
    fig = plt.figure(figsize=(10, 6.2))
    ax = fig.add_subplot(111)
    colors = ['#d62728' if v < 0 else '#2ca02c' for v in values2]
    ax.barh(range(len(labels2)), values2, color=colors, alpha=0.86)
    ax.set_yticks(range(len(labels2)))
    ax.set_yticklabels(labels2)
    ax.axvline(0.0, color='black', linewidth=1.0)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, axis='x', alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)

def plot_box(out_png: str, title: str, groups: List[Tuple[str, List[float]]], ylabel: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    configure_cn_font()
    labels = [g[0] for g in groups]
    data = [g[1] for g in groups]
    fig = plt.figure(figsize=(10, 5.2))
    ax = fig.add_subplot(111)
    try:
        ax.boxplot(data, tick_labels=labels, showfliers=False)
    except TypeError:
        ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)

def main() -> None:
    ap = argparse.ArgumentParser(description='Q3：敏感性分析与假设检验（问题一模型稳定性）')
    ap.add_argument('--outdir', type=str, default='analysis_results', help='输出目录（默认会自动加日期）')
    ap.add_argument('--suite', type=str, default='core', choices=['core', 'extended'], help='场景库规模')
    ap.add_argument('--scenario', type=str, default='短视频_室温', help='选择一个场景做 Q3 分析')
    ap.add_argument('--dt', type=float, default=8.0, help='场景步长 dt（秒）；Q3建议 >=5 加速')
    ap.add_argument('--max-hours', type=float, default=120.0, help='最长仿真时长上限（小时）')
    ap.add_argument('--delta', type=float, default=0.1, help='OAT 扰动比例（如 0.1 表示 ±10%）')
    ap.add_argument('--n', type=int, default=200, help='Monte Carlo 样本数')
    ap.add_argument('--seed', type=int, default=20260130, help='随机种子')
    ap.add_argument('--vary', type=str, default='both', choices=['internal', 'external', 'both'], help='Monte Carlo 采样范围')
    ap.add_argument('--variants', type=str, default=','.join(available_variants()), help='假设变体列表，用逗号分隔')
    ap.add_argument('--bootstrap', type=int, default=800, help='bootstrap 次数（假设检验）')
    ap.add_argument('--no-plots', action='store_true', help='只输出 CSV，不画图')
    args = ap.parse_args()
    if str(args.outdir) == 'analysis_results':
        args.outdir = make_dated_outdir('analysis_results')
    ensure_dir(str(args.outdir))
    dt_s = max(1.0, float(args.dt))
    max_hours = max(0.1, float(args.max_hours))
    n_mc = max(10, int(args.n))
    suite = scenario_suite(dt_s, suite=str(args.suite))
    tup = next((x for x in suite if x[0] == str(args.scenario)), None)
    if tup is None:
        raise SystemExit(f'未找到场景: {args.scenario}。可选: {[x[0] for x in suite]}')
    scenario_name, base_step, _ = tup
    base_params = m.ModelParams()
    step0 = copy.deepcopy(base_step)
    base_metrics = run_metrics(base_params, step0, max_hours=max_hours, extend_until_empty=True)
    baseline_rows = [{'scenario': scenario_name, 't_empty_h': base_metrics.t_empty_h, 'T_max_C': base_metrics.T_max_C, 'avg_p_sys_W': base_metrics.avg_p_sys_W, 'perf_min': base_metrics.perf_min, 'terminated_reason': base_metrics.terminated_reason}]
    write_csv(os.path.join(str(args.outdir), 'q3_baseline.csv'), baseline_rows)
    print(f'[Q3] baseline 完成: {scenario_name} | t_empty_h={base_metrics.t_empty_h:.2f} | T_max={base_metrics.T_max_C:.1f}')
    specs = default_param_specs()
    internal_specs = [s for s in specs if s.kind == 'model']
    oat_rows: List[Dict[str, object]] = []
    delta = max(0.01, float(args.delta))
    for spec in internal_specs:
        nominal = float(get_by_path(base_params, spec.path))
        if nominal > 1e-12:
            v_plus = nominal * (1.0 + delta)
            v_minus = nominal * (1.0 - delta)
            delta_plus = delta
            delta_minus = -delta
        else:
            v_plus = nominal + delta
            v_minus = nominal - delta
            delta_plus = delta
            delta_minus = -delta
        p1 = copy.deepcopy(base_params)
        s1 = copy.deepcopy(step0)
        set_by_path(p1, spec.path, v_plus)
        r_plus = run_metrics(p1, s1, max_hours=max_hours, extend_until_empty=True)
        p2 = copy.deepcopy(base_params)
        s2 = copy.deepcopy(step0)
        set_by_path(p2, spec.path, v_minus)
        r_minus = run_metrics(p2, s2, max_hours=max_hours, extend_until_empty=True)
        S_plus = compute_oat_sensitivity(base_metrics.t_empty_h, r_plus.t_empty_h, delta_plus)
        S_minus = compute_oat_sensitivity(base_metrics.t_empty_h, r_minus.t_empty_h, delta_minus)
        oat_rows.append({'scenario': scenario_name, 'param': spec.name, 'path': spec.path, 'nominal': nominal, 't_empty_base_h': base_metrics.t_empty_h, 't_empty_plus_h': r_plus.t_empty_h, 't_empty_minus_h': r_minus.t_empty_h, 'S_plus': S_plus, 'S_minus': S_minus, 'Tmax_base_C': base_metrics.T_max_C, 'Tmax_plus_C': r_plus.T_max_C, 'Tmax_minus_C': r_minus.T_max_C})
    write_csv(os.path.join(str(args.outdir), 'q3_oat_stability.csv'), oat_rows)
    if not args.no_plots:
        labels = [r['param'] for r in oat_rows]
        vals = [0.5 * (float(r.get('S_plus', 0.0) or 0.0) + float(r.get('S_minus', 0.0) or 0.0)) for r in oat_rows]
        plot_barh(os.path.join(str(args.outdir), 'q3_oat_tornado.png'), f'OAT 一阶灵敏度（续航）| 场景:{scenario_name}', [str(x) for x in labels], [float(x) for x in vals], 'S = (Δ续航/续航)/(Δ参数/参数)')
    rnd = random.Random(int(args.seed))
    if args.vary == 'internal':
        mc_specs = [s for s in specs if s.kind == 'model']
    elif args.vary == 'external':
        mc_specs = [s for s in specs if s.kind == 'step']
    else:
        mc_specs = list(specs)
    base_param_values: Dict[str, float] = {}
    for spec in mc_specs:
        if spec.kind == 'model':
            base_param_values[spec.path] = float(get_by_path(base_params, spec.path))
        else:
            base_param_values[spec.path] = float(get_by_path(step0, spec.path))
    mc_rows: List[Dict[str, object]] = []
    for i in range(n_mc):
        params_i = copy.deepcopy(base_params)
        step_i = copy.deepcopy(step0)
        row: Dict[str, object] = {'scenario': scenario_name, 'sample': i}
        for spec in mc_specs:
            nominal = float(base_param_values[spec.path])
            v = sample_param(rnd, nominal, spec)
            if spec.kind == 'model':
                set_by_path(params_i, spec.path, v)
            else:
                set_by_path(step_i, spec.path, v)
            row[f'x:{spec.name}'] = v
        met = run_metrics(params_i, step_i, max_hours=max_hours, extend_until_empty=True)
        row.update({'t_empty_h': met.t_empty_h, 'T_max_C': met.T_max_C, 'avg_p_sys_W': met.avg_p_sys_W, 'perf_min': met.perf_min, 'terminated_reason': met.terminated_reason})
        mc_rows.append(row)
        if (i + 1) % max(10, n_mc // 10) == 0:
            print(f'[Q3] Monte Carlo: {i + 1}/{n_mc} 完成')
    write_csv(os.path.join(str(args.outdir), 'q3_mc_samples.csv'), mc_rows)
    y_t = [float(r.get('t_empty_h', float('nan')) or float('nan')) for r in mc_rows]
    y_T = [float(r.get('T_max_C', float('nan')) or float('nan')) for r in mc_rows]
    y_p = [float(r.get('avg_p_sys_W', float('nan')) or float('nan')) for r in mc_rows]
    summary_rows = [{'scenario': scenario_name, 'metric': 't_empty_h', 'mean': mean(y_t), 'std': stdev(y_t), 'cv': stdev(y_t) / mean(y_t) if math.isfinite(mean(y_t)) and abs(mean(y_t)) > 1e-12 else float('nan'), 'p05': percentile(y_t, 0.05), 'p50': percentile(y_t, 0.5), 'p95': percentile(y_t, 0.95)}, {'scenario': scenario_name, 'metric': 'T_max_C', 'mean': mean(y_T), 'std': stdev(y_T), 'cv': stdev(y_T) / mean(y_T) if math.isfinite(mean(y_T)) and abs(mean(y_T)) > 1e-12 else float('nan'), 'p05': percentile(y_T, 0.05), 'p50': percentile(y_T, 0.5), 'p95': percentile(y_T, 0.95)}, {'scenario': scenario_name, 'metric': 'avg_p_sys_W', 'mean': mean(y_p), 'std': stdev(y_p), 'cv': stdev(y_p) / mean(y_p) if math.isfinite(mean(y_p)) and abs(mean(y_p)) > 1e-12 else float('nan'), 'p05': percentile(y_p, 0.05), 'p50': percentile(y_p, 0.5), 'p95': percentile(y_p, 0.95)}]
    write_csv(os.path.join(str(args.outdir), 'q3_mc_summary.csv'), summary_rows)
    sens_rows: List[Dict[str, object]] = []
    for spec in mc_specs:
        xs = [float(r.get(f'x:{spec.name}', float('nan')) or float('nan')) for r in mc_rows]
        sens_rows.append({'scenario': scenario_name, 'param': spec.name, 'path': spec.path, 'kind': spec.kind, 'rho_t_empty': spearmanr(xs, y_t), 'rho_T_max': spearmanr(xs, y_T), 'rho_avg_p': spearmanr(xs, y_p)})
    write_csv(os.path.join(str(args.outdir), 'q3_spearman_sensitivity.csv'), sens_rows)
    if not args.no_plots:
        plot_barh(os.path.join(str(args.outdir), 'q3_spearman_t_empty.png'), f'全局敏感性（Spearman）| 续航 | 场景:{scenario_name}', [str(r['param']) for r in sens_rows], [float(r.get('rho_t_empty', 0.0) or 0.0) for r in sens_rows], 'Spearman ρ（参数 vs 续航）')
    variants = [v.strip() for v in str(args.variants).split(',') if v.strip()]
    for v in variants:
        if v not in available_variants():
            raise SystemExit(f'variants 包含未知项: {v}。可选: {available_variants()}')
    if 'baseline' not in variants:
        variants = ['baseline'] + variants
    shared_draws: List[Dict[str, float]] = []
    rnd2 = random.Random(int(args.seed) + 999)
    for _i in range(n_mc):
        d: Dict[str, float] = {}
        for spec in mc_specs:
            nominal = float(base_param_values[spec.path])
            d[spec.name] = sample_param(rnd2, nominal, spec)
        shared_draws.append(d)
    variant_metrics: Dict[str, Dict[str, List[float]]] = {}
    for v in variants:
        variant_metrics[v] = {'t_empty_h': [], 'T_max_C': [], 'avg_p_sys_W': []}
        for i in range(n_mc):
            params_i = copy.deepcopy(base_params)
            step_i = copy.deepcopy(step0)
            for spec in mc_specs:
                vv = float(shared_draws[i][spec.name])
                if spec.kind == 'model':
                    set_by_path(params_i, spec.path, vv)
                else:
                    set_by_path(step_i, spec.path, vv)
            apply_assumption_variant(v, params_i, step_i)
            met = run_metrics(params_i, step_i, max_hours=max_hours, extend_until_empty=True)
            variant_metrics[v]['t_empty_h'].append(met.t_empty_h)
            variant_metrics[v]['T_max_C'].append(met.T_max_C)
            variant_metrics[v]['avg_p_sys_W'].append(met.avg_p_sys_W)
        print(f'[Q3] 变体完成: {v}')
    test_rows: List[Dict[str, object]] = []
    base_key = 'baseline'
    for v in variants:
        if v == base_key:
            continue
        for metric in ['t_empty_h', 'T_max_C', 'avg_p_sys_W']:
            a = variant_metrics[base_key][metric]
            b = variant_metrics[v][metric]
            delta_mean, ci_lo, ci_hi = paired_bootstrap_test(a, b, n_boot=int(args.bootstrap), seed=int(args.seed) + 7)
            p_boot = paired_bootstrap_pvalue(a, b, n_boot=int(args.bootstrap), seed=int(args.seed) + 17)
            p_mwu = mann_whitney_u_pvalue(a, b)
            test_rows.append({'scenario': scenario_name, 'metric': metric, 'H0': '变体与baseline输出无差异', 'variant': v, 'delta_mean(variant-baseline)': delta_mean, 'ci95_low': ci_lo, 'ci95_high': ci_hi, 'p_bootstrap': p_boot, 'p_mann_whitney': p_mwu, 'n': n_mc})
    write_csv(os.path.join(str(args.outdir), 'q3_hypothesis_tests.csv'), test_rows)
    if not args.no_plots and len(variants) >= 2:
        groups = [(v, variant_metrics[v]['t_empty_h']) for v in variants]
        plot_box(os.path.join(str(args.outdir), 'q3_variants_box_t_empty.png'), f'建模假设变体对续航的影响 | 场景:{scenario_name}', groups, 't_empty_h')
    print(f'[Q3] 已输出目录: {os.path.abspath(str(args.outdir))}')
if __name__ == '__main__':
    main()
