from __future__ import annotations
import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Series:
    t_s: List[float]
    cols: Dict[str, List[float]]

def read_csv_series(path: str) -> Series:
    with open(path, 'r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        t_s: List[float] = []
        cols: Dict[str, List[float]] = {}
        for row in reader:
            t = float(row['t_s'])
            t_s.append(t)
            for k, v in row.items():
                if k == 't_s':
                    continue
                cols.setdefault(k, []).append(float(v))
    return Series(t_s=t_s, cols=cols)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def plot_from_csv(csv_path: str, outdir: str, *, dpi: int=160, title_prefix: str='', xlim_h: float | None=None) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except Exception as e:
        raise SystemExit(f'matplotlib 未安装或不可用。请先安装：pip install matplotlib\n原始错误: {e}')

    def configure_cn_font() -> None:
        candidates = ['Microsoft YaHei', 'Microsoft YaHei UI', 'SimHei', 'SimSun', 'DengXian', 'PingFang SC', 'Noto Sans CJK SC', 'Source Han Sans SC']
        available = {f.name for f in font_manager.fontManager.ttflist}
        chosen = next((name for name in candidates if name in available), None)
        if chosen is not None:
            plt.rcParams['font.sans-serif'] = [chosen] + list(plt.rcParams.get('font.sans-serif', []))
        plt.rcParams['axes.unicode_minus'] = False
    configure_cn_font()
    s = read_csv_series(csv_path)
    ensure_dir(outdir)
    t_h = [x / 3600.0 for x in s.t_s]
    t_end_h = t_h[-1] if t_h else 0.0

    def col(name: str, default: float | None=None) -> List[float]:
        if name in s.cols:
            return s.cols[name]
        if default is None:
            raise KeyError(f'CSV 缺少列: {name}')
        return [default for _ in s.t_s]
    soc = col('soc')
    vterm = col('V_term_V')
    current = col('I_A')
    v_avg = sum(vterm) / len(vterm) if vterm else float('nan')
    i_avg = sum(current) / len(current) if current else float('nan')
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(111)
    ax1.plot(t_h, [z * 100 for z in soc], label='SoC (%)', color='#1f77b4', linewidth=2)
    ax1.set_xlabel('时间 (小时)')
    ax1.set_ylabel('SoC (%)', color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(t_h, vterm, label=f'V_term (V) avg={v_avg:.3f}', color='#ff7f0e', linewidth=1.6)
    ax2.plot(t_h, current, label=f'I (A) avg={i_avg:.3f}', color='#2ca02c', linewidth=1.2, alpha=0.8)
    ax2.set_ylabel('电压/电流')
    if xlim_h is not None:
        ax1.set_xlim(0.0, float(xlim_h))
    if title_prefix:
        ax1.set_title(f'{title_prefix} | t_end={t_end_h:.2f}h')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, '01_soc_voltage_current.png'), dpi=dpi)
    plt.close(fig)
    T = col('T_C')
    perf = col('perf_factor')
    T_max = max(T) if T else float('nan')
    perf_min = min(perf) if perf else float('nan')
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(111)
    ax1.plot(t_h, T, label='温度 T (°C)', color='#d62728', linewidth=2)
    ax1.set_xlabel('时间 (小时)')
    ax1.set_ylabel('温度 (°C)')
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(t_h, perf, label=f'perf_factor min={perf_min:.3f}', color='#9467bd', linewidth=1.8)
    ax2.set_ylabel('perf_factor')
    ax2.set_ylim(0.0, 1.05)
    if xlim_h is not None:
        ax1.set_xlim(0.0, float(xlim_h))
    if title_prefix:
        ax1.set_title(f'{title_prefix} | T_max={T_max:.1f}°C')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, '02_temperature_throttle.png'), dpi=dpi)
    plt.close(fig)
    p_sys = col('p_sys_W')
    p_screen = col('p_screen_W', 0.0)
    p_compute = col('p_compute_W', 0.0)
    p_radio = col('p_radio_W', 0.0)
    p_bg = col('p_bg_W', 0.0)
    p_gps = col('p_gps_W', 0.0)
    p_bt = col('p_bt_W', 0.0)
    p_render = col('p_render_extra_W', 0.0)
    p_radio_bg = col('p_radio_bg_extra_W', 0.0)
    p_leak = col('p_thermal_leak_W', 0.0)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    labels = ['screen', 'compute', 'radio', 'background', 'gps', 'bt', 'coupling:render', 'coupling:radio×bg', 'coupling:thermal']
    series = [p_screen, p_compute, p_radio, p_bg, p_gps, p_bt, p_render, p_radio_bg, p_leak]
    ax.stackplot(t_h, series, labels=labels, alpha=0.85)
    ax.plot(t_h, p_sys, color='black', linewidth=1.2, label='total p_sys')
    ax.set_xlabel('时间 (小时)')
    ax.set_ylabel('功耗 (W)')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper right', ncol=2, fontsize=9)
    if xlim_h is not None:
        ax.set_xlim(0.0, float(xlim_h))
    if title_prefix:
        p_avg = sum(p_sys) / len(p_sys) if p_sys else float('nan')
        ax.set_title(f'{title_prefix} | avg_p_sys={p_avg:.2f}W')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, '03_power_breakdown.png'), dpi=dpi)
    plt.close(fig)
    radio_state = col('radio_state')
    fig = plt.figure(figsize=(10, 3.6))
    ax = fig.add_subplot(111)
    ax.step(t_h, radio_state, where='post', color='#1f77b4', linewidth=1.8)
    ax.set_xlabel('时间 (小时)')
    ax.set_ylabel('Radio state (0=IDLE,1=TAIL,2=ACTIVE)')
    ax.set_yticks([0, 1, 2])
    ax.grid(True, alpha=0.25)
    if xlim_h is not None:
        ax.set_xlim(0.0, float(xlim_h))
    if title_prefix:
        ax.set_title(title_prefix)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, '04_radio_state.png'), dpi=dpi)
    plt.close(fig)

def main() -> None:
    ap = argparse.ArgumentParser(description='Plot figures from sim_output.csv')
    ap.add_argument('--csv', type=str, default='sim_output.csv', help='Input CSV path')
    ap.add_argument('--outdir', type=str, default='figures', help='Output directory')
    ap.add_argument('--dpi', type=int, default=160, help='Figure DPI')
    args = ap.parse_args()
    plot_from_csv(args.csv, args.outdir, dpi=int(args.dpi))
    print(f'已生成图像到目录: {os.path.abspath(args.outdir)}')
if __name__ == '__main__':
    main()
