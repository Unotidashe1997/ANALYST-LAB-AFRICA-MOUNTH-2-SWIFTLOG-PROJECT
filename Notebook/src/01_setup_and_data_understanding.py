# %% [md]
# # Week 5 — Data Science Track · Mission DS05
# ## Supply Chain Intelligence & Feature Engineering for Predictive Logistics
# **Client:** SwiftLog Logistics Limited · **Consultancy:** AnalystLab Africa Consulting · **Role:** Junior Data Scientist
#
# This notebook covers all eight parts of the brief: data acquisition, cleaning, EDA (15+ charts), statistics (descriptive + hypothesis tests),
# feature engineering, segmentation, business insights and recommendations.
#
# **Reading guide.** Every chart prints its *interpretation* and *business implication* directly beneath it. All numbers in the text are computed from the data
# by the code above them (no hand-typed figures), so re-running the notebook keeps text and results consistent.
#
# **Honest-scope note.** Several items in the brief cannot be computed from the supplied data (e.g. there is no supplier key on any table, no warehouse capacity, no ship/delivery dates).
# Where that happens the notebook says so, shows the evidence, and either uses a clearly labelled proxy or reports the gap — it never invents data.

# %%
import os, re, json, base64, io, shutil, itertools, textwrap, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 40); pd.set_option('display.width', 200); pd.set_option('display.float_format', lambda v: f'{v:,.3f}')

BASE = Path.cwd().parent if Path.cwd().name == 'Notebook' else Path.cwd()
RAW, CLEAN = BASE/'Data'/'Original_Datasets', BASE/'Data'/'Cleaned_Datasets'
CHARTS, DASH, REPORTS = BASE/'Visualizations'/'Charts', BASE/'Visualizations'/'Dashboards', BASE/'Reports'
for p in (CLEAN, CHARTS, DASH, REPORTS): p.mkdir(parents=True, exist_ok=True)

# Palette (validated reference palette, categorical slots 1-3 + neutrals)
BLUE, ORANGE, AQUA, GREY, INK = '#2a78d6', '#eb6834', '#1baf7a', '#9a9993', '#0b0b0b'
plt.rcParams.update({'figure.dpi': 100, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e6e3',
    'grid.linewidth': .8, 'axes.axisbelow': True, 'axes.edgecolor': '#b8b8b3', 'axes.titlesize': 12.5, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e', 'font.size': 10, 'legend.frameon': False})

R = {}      # results store (feeds the report + deck)
CH = []     # chart registry
def chart(name, title, interp, impl):
    """Save the current figure and print its interpretation + business implication."""
    fig = plt.gcf(); fig.savefig(CHARTS/f'{name}.png', dpi=150, bbox_inches='tight', facecolor='white')
    CH.append(dict(file=f'{name}.png', title=title, interp=interp, impl=impl))
    print(f'Chart {len(CH)} · {title}\n  Interpretation: {interp}\n  Business implication: {impl}')
def short(x):
    x = float(x); a = abs(x)
    return f'{x/1e6:,.2f}M' if a >= 1e6 else f'{x/1e3:,.1f}K' if a >= 1e4 else f'{x:,.0f}'
def pct(x, d=1): return f'{100*x:.{d}f}%'
print('Environment ready. Base folder:', BASE.name)

# %% [md]
# ---
# # PART 1 — Data Acquisition & Understanding

# %%
NAMES = ['Customers','Orders','Order_Items','Products','Suppliers','Warehouses','Inventory','Shipments','Delivery_Performance','Employees','Returns','Data_Dictionary']
raw = {k: pd.read_csv(RAW/f'SwiftLog_{k}.csv', encoding='utf-8-sig') for k in NAMES}   # utf-8-sig removes the byte-order mark present in some files
overview = pd.DataFrame({k: {'rows': len(d), 'columns': d.shape[1], 'column names': ', '.join(d.columns), 'missing cells': int(d.isna().sum().sum()),
                             'exact duplicate rows': int(d.duplicated().sum())} for k, d in raw.items()}).T
R['n_tables'] = len(raw); R['total_rows'] = int(sum(len(d) for k, d in raw.items() if k != 'Data_Dictionary'))
overview

# %%
# Data types actually inferred on import, per table
dtypes = pd.DataFrame([(t, c, str(d[c].dtype)) for t, d in raw.items() if t != 'Data_Dictionary' for c in d.columns], columns=['table', 'column', 'inferred dtype'])
dtypes

# %% [md]
# ### Primary and foreign keys — declared, then verified against the data
# The key map below is declared from the column names and then **tested** (uniqueness for primary keys, orphan counts for foreign keys).

# %%
PK = {'Customers': ['CustomerID'], 'Orders': ['OrderID'], 'Order_Items': ['OrderItemID'], 'Products': ['ProductID'], 'Suppliers': ['SupplierID'],
      'Warehouses': ['WarehouseID'], 'Inventory': ['WarehouseID','ProductID'], 'Shipments': ['ShipmentID'], 'Delivery_Performance': ['ShipmentID'],
      'Employees': ['EmployeeID'], 'Returns': ['ReturnID']}
pk_rows = [dict(table=t, primary_key=' + '.join(k), unique=not raw[t].duplicated(k).any(), duplicate_keys=int(raw[t].duplicated(k).sum())) for t, k in PK.items()]
pk_table = pd.DataFrame(pk_rows)
FK = [('Orders','CustomerID','Customers'), ('Order_Items','OrderID','Orders'), ('Order_Items','ProductID','Products'), ('Inventory','WarehouseID','Warehouses'),
      ('Inventory','ProductID','Products'), ('Shipments','OrderID','Orders'), ('Shipments','WarehouseID','Warehouses'), ('Delivery_Performance','ShipmentID','Shipments'),
      ('Employees','WarehouseID','Warehouses'), ('Returns','OrderID','Orders')]
fk_table = pd.DataFrame([dict(child=c, foreign_key=col, parent=p, orphan_rows=int((~raw[c][col].isin(raw[p][PK[p][0]])).sum()),
                              child_rows=len(raw[c]), parent_keys_unused=int((~raw[p][PK[p][0]].isin(raw[c][col])).sum())) for c, col, p in FK])
print('PRIMARY KEYS'); print(pk_table.to_string(index=False)); print('\nFOREIGN KEYS'); print(fk_table.to_string(index=False))
R['inventory_dup_keys'] = int(raw['Inventory'].duplicated(PK['Inventory']).sum())
R['fk_orphans_total'] = int(fk_table.orphan_rows.sum())
print(f"\nFinding: all foreign keys resolve (orphans = {R['fk_orphans_total']}), so the tables join cleanly. "
      f"Inventory is NOT unique on (WarehouseID, ProductID): {R['inventory_dup_keys']} repeated keys — handled in Part 2.")

# %%
# Summary statistics: numeric and categorical columns
num_cols = {'Order_Items': ['Quantity'], 'Products': ['UnitPrice'], 'Inventory': ['Stock'], 'Shipments': ['DelayDays']}
summ = pd.concat({f'{t}.{c}': raw[t][c].describe() for t, cs in num_cols.items() for c in cs}, axis=1).T
print('Numeric summary'); print(summ.to_string())
cat_cols = {'Orders': ['Status'], 'Customers': ['City'], 'Products': ['Category'], 'Warehouses': ['Location'], 'Returns': ['Reason'], 'Delivery_Performance': ['Status']}
for t, cs in cat_cols.items():
    for c in cs: print(f'\n{t}.{c}:', raw[t][c].value_counts().to_dict())
print('\nOrder date range:', pd.to_datetime(raw['Orders'].OrderDate, format='%m/%d/%Y').agg(['min','max']).dt.date.tolist())

# %% [md]
# ### Data-quality assessment
# Beyond nulls and duplicates (both near-zero), we tested **business-logic consistency** across tables. These checks drive the cleaning rules in Part 2.

# %%
o, oi, sh, dp, rt, inv = raw['Orders'].copy(), raw['Order_Items'], raw['Shipments'], raw['Delivery_Performance'], raw['Returns'], raw['Inventory']
o['d'] = pd.to_datetime(o.OrderDate, format='%m/%d/%Y')
shx = sh.merge(o[['OrderID','Status']], on='OrderID'); rtx = rt.merge(o[['OrderID','Status']], on='OrderID')
dpm = sh.merge(dp, on='ShipmentID', suffixes=('_sh','_dp'))
sept_orders = int((o.d.dt.month == 9).sum())
checks = [
 ('Missing values (all tables)', int(sum(d.isna().sum().sum() for d in raw.values())), 'None found'),
 ('Exact duplicate rows — Inventory', int(inv.duplicated().sum()), 'Remove'),
 ('Inventory rows sharing a (Warehouse, Product) key with another row', int(inv.duplicated(PK['Inventory'], keep=False).sum()), 'Treat as stock lots and sum (documented assumption)'),
 ('Order_Items lines repeating an (Order, Product) pair', int(oi.duplicated(['OrderID','ProductID']).sum()), 'Consolidate quantities'),
 ('Orders with no line items', int((~o.OrderID.isin(oi.OrderID)).sum()), 'Keep, flag (value = 0)'),
 ('Shipments recorded for CANCELLED orders', int((shx.Status=='Cancelled').sum()), 'Keep for operational analysis, flag; sensitivity-test'),
 ('Shipments recorded for PENDING orders', int((shx.Status=='Pending').sum()), 'Keep, flag'),
 ('Returns recorded against non-Delivered orders', int((rtx.Status!='Delivered').sum()), 'Keep, flag; report as integrity issue'),
 ('Orders with more than one return record', int(rt.OrderID.duplicated().sum()), 'Keep (multi-reason returns)'),
 ('Shipments.DelayDays ≠ Delivery_Performance.DelayDays', int((dpm.DelayDays_sh != dpm.DelayDays_dp).sum()), 'None — tables agree'),
 ('Delivery status is a pure rule of DelayDays (≤2 On Time, ≥3 Delayed)', int(((dp.DelayDays>=3) != (dp.Status=='Delayed')).sum()) , '0 exceptions → status is derived, not independent information'),
 ('Orders placed in the partial month (Sep 2026, data ends 8 Sep)', sept_orders, 'Exclude Sep from month-over-month comparisons'),
 ('Supplier key present on any operational table', 0, 'DATA GAP — supplier analysis cannot be linked to deliveries'),
 ('Warehouse capacity field', 0, 'DATA GAP — utilisation can only be proxied'),
 ('Ship / delivery dates (only OrderDate exists)', 0, 'DATA GAP — delivery duration cannot be computed; DelayDays used'),
 ('Data dictionary column-level definitions', 0, 'DATA GAP — dictionary lists file names only'),
]
dq = pd.DataFrame(checks, columns=['check', 'count', 'treatment / finding'])
R['dq'] = {c: int(n) for c, n, _ in checks}
dq

# %% [md]
# ### Data relationship diagram
# How the 11 datasets connect. Solid arrows = verified foreign keys (parent ← child). **Suppliers has no key linking it to anything.**

# %%
fig, ax = plt.subplots(figsize=(15, 8.2)); ax.set_xlim(0, 15); ax.set_ylim(0, 8.6); ax.axis('off'); ax.grid(False)
pos = {'Customers': (1.7, 7), 'Orders': (5.4, 7), 'Order_Items': (9.1, 7), 'Products': (12.9, 7),
       'Returns': (1.7, 4), 'Shipments': (5.4, 4), 'Delivery_Performance': (9.1, 4), 'Inventory': (12.9, 4),
       'Employees': (3.2, 1), 'Warehouses': (7.6, 1), 'Suppliers': (12.9, 1)}
keys = {'Customers': 'CustomerID (PK)', 'Orders': 'OrderID (PK) · CustomerID (FK)', 'Order_Items': 'OrderItemID (PK)\nOrderID, ProductID (FK)', 'Products': 'ProductID (PK)',
        'Returns': 'ReturnID (PK) · OrderID (FK)', 'Shipments': 'ShipmentID (PK)\nOrderID, WarehouseID (FK)', 'Delivery_Performance': 'ShipmentID (PK/FK)', 'Inventory': 'WarehouseID, ProductID (FK)\n(not unique)',
        'Employees': 'EmployeeID (PK) · WarehouseID (FK)', 'Warehouses': 'WarehouseID (PK)', 'Suppliers': 'SupplierID (PK)\nNO LINK TO ANY TABLE'}
bw, bh = 3.1, 1.25
for t, (x, y) in pos.items():
    dashed = t == 'Suppliers'
    ax.add_patch(FancyBboxPatch((x-bw/2, y-bh/2), bw, bh, boxstyle='round,pad=0.02,rounding_size=0.12', fc='#fdecea' if dashed else '#eaf2fc', ec=ORANGE if dashed else BLUE, lw=1.6, ls='--' if dashed else '-'))
    ax.text(x, y+0.28, f'{t}  ({len(raw[t]):,})', ha='center', va='center', fontsize=10.5, fontweight='bold', color=INK)
    ax.text(x, y-0.2, keys[t], ha='center', va='center', fontsize=8, color='#52514e')
edges = [('Customers','Orders'), ('Orders','Order_Items'), ('Products','Order_Items'), ('Orders','Shipments'), ('Orders','Returns'), ('Shipments','Delivery_Performance'),
         ('Products','Inventory'), ('Warehouses','Inventory'), ('Warehouses','Shipments'), ('Warehouses','Employees')]
def edge_pts(a, b):
    (x1, y1), (x2, y2) = pos[a], pos[b]
    if abs(y1-y2) < .1: s = 1 if x2 > x1 else -1; return (x1+s*bw/2, y1), (x2-s*bw/2, y2)
    if abs(x1-x2) < .1: s = 1 if y2 > y1 else -1; return (x1, y1+s*bh/2), (x2, y2-s*bh/2)
    s = 1 if y2 > y1 else -1; return (x1, y1+s*bh/2), (x2, y2-s*bh/2)
for p, c in edges:
    (a, b) = edge_pts(p, c)
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=14, lw=1.4, color='#52514e', shrinkA=0, shrinkB=0))
ax.text(7.5, 8.35, 'SwiftLog data model — 11 datasets, parent → child (1 : many)', ha='center', fontsize=13, fontweight='bold', color=INK)
chart('01_data_relationship_diagram', 'Data relationship diagram',
      'Orders is the hub: customers, line items, shipments and returns all hang off OrderID. Warehouses connect shipments, stock and staff. Suppliers is isolated.',
      'Operational questions (delivery, returns, customer value) can be joined end-to-end, but supplier performance cannot — SwiftLog must add a SupplierID to Products or Shipments to evaluate suppliers.')
