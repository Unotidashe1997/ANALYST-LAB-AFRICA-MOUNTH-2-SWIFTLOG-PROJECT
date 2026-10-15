"""Run AFTER run_pipeline.py (needs Notebook/results.json). From the project root: python Notebook/src/07_build_dashboards.py"""
import json, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
BLUE, ORANGE, AQUA, GREY, INK = '#2a78d6', '#eb6834', '#1baf7a', '#9a9993', '#0b0b0b'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e6e3', 'axes.axisbelow': True, 'axes.titleweight': 'bold', 'axes.titlesize': 11, 'axes.labelsize': 9, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5})
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; P = str(ROOT) + '/'; R = json.load(open(str(ROOT / 'Notebook' / 'results.json')))['R']
def short(v): return f'{v/1e6:.2f}M' if abs(v) >= 1e6 else f'{v/1e3:.1f}K' if abs(v) >= 1e3 else f'{v:.0f}'
def tile(ax, label, value, sub): ax.axis('off'); ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc='#f4f6fa', ec='#d9dee8')); ax.text(.06, .72, label, fontsize=9, color='#52514e', transform=ax.transAxes); ax.text(.06, .36, value, fontsize=16, fontweight='bold', color=INK, transform=ax.transAxes); ax.text(.06, .1, sub, fontsize=8, color='#52514e', transform=ax.transAxes)
# ---- Dashboard 1: Executive KPI
fig = plt.figure(figsize=(16, 10)); gs = fig.add_gridspec(4, 6, height_ratios=[.7, 1.5, 1.5, .02], hspace=.5, wspace=.9)
fig.suptitle('SwiftLog Logistics — Executive KPI Dashboard (orders 1 Jan – 8 Sep 2026)', fontsize=15, fontweight='bold', x=.06, ha='left', y=.97)
k = [('Confirmed order value', short(R['rev_confirmed']), f"gross {short(R['rev_gross'])}"), ('Delivered share of orders', f"{R['delivered_rate']*100:.1f}%", f"cancelled {R['cancel_rate']*100:.1f}% · pending {R['pending_rate']*100:.1f}%"), ('Shipments delayed (≥3 d)', f"{R['delay_rate']*100:.1f}%", f"mean delay {R['mean_delay']:.2f} days"),
     ('Orders with a return', f"{R['return_rate']*100:.1f}%", f"{R['n_returns']} return records"), ('Orders / customers', f"{R['n_orders']:,} / {R['n_customers_total']:,}", f"{R['n_customers_ordering']:,} customers ordered"), ('Low-stock positions', f"{R['low_stock_positions']}", f"of {R['inv_positions']:,} (<20 units)")]
for i, (a, b, c) in enumerate(k): tile(fig.add_subplot(gs[0, i]), a, b, c)
mo = pd.Series(R['monthly_revenue']); ax = fig.add_subplot(gs[1, :3]); lab = list(mo.keys()); x = np.arange(len(lab))
for key, c, nm in [('gross', GREY, 'Gross'), ('confirmed', BLUE, 'Confirmed'), ('realised', ORANGE, 'Delivered')]:
    v = np.array([mo[m][key] for m in lab]) / 1e6; ax.plot(x[:-1], v[:-1], color=c, lw=2.2, marker='o', ms=5, label=nm)
ax.set_xticks(x[:-1]); ax.set_xticklabels(lab[:-1], rotation=0, fontsize=7.5); ax.set_ylim(0, 3.4); ax.set_title('Monthly order value (Jan–Aug; Sep excluded, partial)'); ax.set_ylabel('Millions'); ax.set_xlabel('Month'); ax.legend(ncol=3, fontsize=8, loc='upper left')
ax = fig.add_subplot(gs[1, 3:]); cr = R['category_revenue']; names = list(cr); ax.bar(names, [cr[n]['revenue'] / 1e6 for n in names], color=BLUE, width=.55)
for i, n in enumerate(names): ax.text(i, cr[n]['revenue'] / 1e6 + .1, f"{cr[n]['share']*100:.0f}%", ha='center', fontsize=9)
ax.set_title('Confirmed revenue by category'); ax.set_ylabel('Millions'); ax.set_xlabel('Category'); ax.set_ylim(0, 6.5)
ax = fig.add_subplot(gs[2, :2]); st = R['status_table']; ss = ['Delivered', 'Pending', 'Cancelled']; ax.bar(ss, [st[s]['n'] for s in ss], color=[BLUE, ORANGE, GREY], width=.55)
for i, s in enumerate(ss): ax.text(i, st[s]['n'] + 25, f"{int(st[s]['n']):,}", ha='center', fontsize=9)
ax.set_title('Orders by status'); ax.set_ylabel('Orders'); ax.set_xlabel('Status'); ax.set_ylim(0, 2000)
ax = fig.add_subplot(gs[2, 2:4]); sg = sorted(R['segments'].values(), key=lambda v: -v['share_of_confirmed_revenue']); import textwrap
ax.barh([textwrap.fill(v['segment'], 24) for v in sg][::-1], [v['share_of_confirmed_revenue'] * 100 for v in sg][::-1], color=AQUA, height=.55); ax.set_title('Customer segments: % of revenue'); ax.set_xlabel('% of confirmed revenue'); ax.tick_params(axis='y', labelsize=7.5)
ax = fig.add_subplot(gs[2, 4:]); rr = R['return_reasons']; ax.barh(list(rr)[::-1], list(rr.values())[::-1], color=BLUE, height=.5)
for i, v in enumerate(list(rr.values())[::-1]): ax.text(v + 3, i, v, va='center', fontsize=9)
ax.set_title('Return reasons'); ax.set_xlabel('Records'); ax.set_xlim(0, 230)
fig.text(.06, .02, 'Reading guide: "confirmed" excludes cancelled orders; "delivered" counts only status = Delivered. Currency units as supplied (not named in the data dictionary).', fontsize=8.5, color='#52514e')
fig.savefig(P + 'Visualizations/Dashboards/Dashboard_1_Executive_KPIs.png', dpi=130, bbox_inches='tight', facecolor='white'); plt.close(fig)
# ---- Dashboard 2: Operations
wl = R['wh_load']; wd = R['wh_delay']; wi = R['wh_inventory']; ws = sorted(wl); fig = plt.figure(figsize=(16, 9.5)); gs = fig.add_gridspec(2, 2, hspace=.45, wspace=.18)
fig.suptitle('SwiftLog Logistics — Warehouse & Delivery Operations Dashboard', fontsize=15, fontweight='bold', x=.06, ha='left', y=.98)
ax = fig.add_subplot(gs[0, 0]); v = [wl[w]['ship_per_emp'] for w in ws]; m = np.mean(v); ax.bar(ws, v, color=[ORANGE if a > 1.25 * m else BLUE for a in v], width=.65); ax.axhline(m, color=INK, ls='--', lw=1); ax.set_title('Shipments per employee (orange = >25% above network mean)'); ax.set_xlabel('Warehouse'); ax.set_ylabel('Shipments ÷ staff')
ax = fig.add_subplot(gs[0, 1]); v = [wd[w]['rate'] * 100 for w in ws]; ax.bar(ws, v, color=BLUE, width=.65); ax.axhline(R['delay_rate'] * 100, color=ORANGE, lw=2, ls='--'); ax.text(14.4, R['delay_rate'] * 100 + 1.5, f"network {R['delay_rate']*100:.1f}%", ha='right', color=ORANGE, fontsize=9); ax.set_ylim(0, 85); ax.set_title(f"% shipments delayed (differences not significant, p = {R['p_delay_wh']:.2f})"); ax.set_xlabel('Warehouse'); ax.set_ylabel('% delayed')
ax = fig.add_subplot(gs[1, 0]); v = [wi[w]['low'] for w in ws]; ax.bar(ws, v, color=[ORANGE if a >= 5 else BLUE for a in v], width=.65); ax.set_title('Low-stock positions (<20 units; orange = 5 or more)'); ax.set_xlabel('Warehouse'); ax.set_ylabel('Positions')
for i, a in enumerate(v): ax.text(i, a + .12, int(a), ha='center', fontsize=8.5)
ax = fig.add_subplot(gs[1, 1]); rd = {k.split('|')[1]: x for k, x in R['region_delay'].items() if k.startswith('Customer')}; ct = sorted(rd, key=lambda k: -rd[k]['rate']); ax.bar(ct, [rd[c]['rate'] * 100 for c in ct], color=AQUA, width=.5); ax.axhline(R['delay_rate'] * 100, color=ORANGE, lw=2, ls='--')
for i, c in enumerate(ct): ax.text(i, rd[c]['rate'] * 100 + 1.2, f"{rd[c]['rate']*100:.1f}%", ha='center', fontsize=9)
ax.set_ylim(0, 85); ax.set_title(f"% delayed by destination city (p = {R['p_delay_custcity']:.2f})"); ax.set_xlabel('Customer city'); ax.set_ylabel('% delayed')
fig.savefig(P + 'Visualizations/Dashboards/Dashboard_2_Operations.png', dpi=130, bbox_inches='tight', facecolor='white'); plt.close(fig)
