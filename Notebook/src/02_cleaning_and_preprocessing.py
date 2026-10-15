# %% [md]
# ---
# # PART 2 — Data Cleaning & Preprocessing
# All cleaning lives in **reusable functions** (`standardise_text`, `parse_dates`, `iqr_flag`, `clean_all`, `build_order_master`) so the same pipeline can be re-run on new extracts.
# Every action is written to a transformation log (`LOG`) which is displayed at the end of this part.

# %%
LOG = []
def log(step, table, action, rows, why): LOG.append(dict(step=len(LOG)+1, table=table, action=action, rows_affected=int(rows), rationale=why))

def standardise_text(df, table, title_cols=(), id_cols=()):
    """Trim + collapse whitespace on all text columns; Title-case categorical columns; upper-case ID columns."""
    df = df.copy(); changed = 0
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith('str'):
            s0 = df[c].astype(str); s = s0.str.strip().str.replace(r'\s+', ' ', regex=True)
            if c in title_cols: s = s.str.title()
            if c in id_cols: s = s.str.upper()
            changed += int((s != s0).sum()); df[c] = s
    log('Standardise text', table, 'trim whitespace, unify case of categorical/ID columns', changed, 'Prevents duplicate categories such as "lagos" vs "Lagos"')
    return df

def parse_dates(df, col, fmt):
    df = df.copy(); df[col] = pd.to_datetime(df[col], format=fmt, errors='coerce')
    bad = int(df[col].isna().sum()); log('Convert data type', 'Orders', f'{col} text → datetime ({fmt})', len(df), f'Unparseable dates: {bad}')
    return df

def iqr_flag(s, k=1.5):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1; lo, hi = q1 - k*i, q3 + k*i
    return (s < lo) | (s > hi), lo, hi

LOW_STOCK = 20      # assumption: <20 units at a warehouse-product position = low stock (≈4% of the 0-500 range)
def clean_all(raw):
    cl = {}
    cl['Customers'] = standardise_text(raw['Customers'], 'Customers', title_cols=['City'], id_cols=['CustomerID'])
    cl['Products'] = standardise_text(raw['Products'], 'Products', title_cols=['Category'], id_cols=['ProductID'])
    cl['Suppliers'] = standardise_text(raw['Suppliers'], 'Suppliers', id_cols=['SupplierID'])
    cl['Warehouses'] = standardise_text(raw['Warehouses'], 'Warehouses', title_cols=['Location'], id_cols=['WarehouseID'])
    cl['Employees'] = standardise_text(raw['Employees'], 'Employees', id_cols=['EmployeeID', 'WarehouseID'])
    cl['Returns'] = standardise_text(raw['Returns'], 'Returns', title_cols=['Reason'], id_cols=['ReturnID', 'OrderID'])
    # Orders
    o = standardise_text(raw['Orders'], 'Orders', title_cols=['Status'], id_cols=['OrderID', 'CustomerID'])
    o = parse_dates(o, 'OrderDate', '%m/%d/%Y')
    o['Year'], o['Month'], o['Quarter'] = o.OrderDate.dt.year, o.OrderDate.dt.month, o.OrderDate.dt.quarter
    o['MonthName'], o['YearMonth'], o['DayOfWeek'] = o.OrderDate.dt.strftime('%b'), o.OrderDate.dt.strftime('%Y-%m'), o.OrderDate.dt.day_name()
    last = o.OrderDate.max(); o['IsPartialMonth'] = (o.OrderDate.dt.to_period('M') == last.to_period('M')) & (last != last + pd.offsets.MonthEnd(0))
    log('Create date features', 'Orders', 'Year, Month, Quarter, MonthName, YearMonth, DayOfWeek, IsPartialMonth', len(o), f'Last order is {last.date()}, so {last.strftime("%b %Y")} is a partial month and is excluded from trend comparisons')
    cl['Orders'] = o
    # Order items: consolidate repeated (Order, Product) lines, attach price/category
    oi = standardise_text(raw['Order_Items'], 'Order_Items', id_cols=['OrderItemID', 'OrderID', 'ProductID'])
    n_rep = int(oi.duplicated(['OrderID', 'ProductID']).sum())
    oi = oi.groupby(['OrderID', 'ProductID'], as_index=False).agg(OrderItemID=('OrderItemID', 'first'), Quantity=('Quantity', 'sum'), n_merged=('OrderItemID', 'size'))
    log('Consolidate duplicates', 'Order_Items', 'merged repeated (OrderID, ProductID) lines by summing Quantity', n_rep, 'Same product ordered twice in one order = one basket line; avoids double counting basket size')
    oi = oi.merge(cl['Products'][['ProductID', 'Category', 'UnitPrice']], on='ProductID', how='left'); oi['LineValue'] = (oi.Quantity * oi.UnitPrice).round(2)
    log('Create calculated column', 'Order_Items', 'LineValue = Quantity × UnitPrice (joined from Products)', len(oi), 'Revenue is not stored anywhere in the raw data')
    cl['Order_Items'] = oi
    # Inventory: drop exact duplicates, then consolidate duplicate keys as stock lots
    inv = standardise_text(raw['Inventory'], 'Inventory', id_cols=['WarehouseID', 'ProductID'])
    n_ex = int(inv.duplicated().sum()); inv = inv.drop_duplicates(); log('Remove duplicates', 'Inventory', 'dropped exact duplicate rows', n_ex, 'Identical rows are repeated records')
    n_key = int(inv.duplicated(['WarehouseID', 'ProductID']).sum())
    inv = inv.groupby(['WarehouseID', 'ProductID'], as_index=False).agg(Stock=('Stock', 'sum'), n_lots=('Stock', 'size'))
    inv['LowStock'] = inv.Stock < LOW_STOCK
    log('Consolidate duplicates', 'Inventory', 'summed Stock over repeated (WarehouseID, ProductID) keys → n_lots; added LowStock flag (<20)', n_key, 'ASSUMPTION: repeated keys with different quantities are separate stock lots. Sensitivity test below')
    cl['Inventory'] = inv
    # Shipments + delivery performance
    sh = standardise_text(raw['Shipments'], 'Shipments', id_cols=['ShipmentID', 'OrderID', 'WarehouseID'])
    dp = standardise_text(raw['Delivery_Performance'], 'Delivery_Performance', title_cols=['Status'], id_cols=['ShipmentID'])
    sh = sh.merge(dp[['ShipmentID', 'Status']].rename(columns={'Status': 'DeliveryStatus'}), on='ShipmentID', how='left')
    sh['IsDelayed'] = (sh.DelayDays >= 3).astype(int)
    sh['DelaySeverity'] = pd.cut(sh.DelayDays, [-1, 0, 2, 4, 7], labels=['On schedule (0)', 'Minor (1-2)', 'Moderate (3-4)', 'Severe (5-7)']).astype(str)
    mism = int(((sh.IsDelayed == 1) != (sh.DeliveryStatus == 'Delayed')).sum())
    log('Merge + conditional column', 'Shipments', 'joined Delivery_Performance status; IsDelayed = DelayDays ≥ 3; DelaySeverity bands', len(sh), f'Rule verified against the supplied Status field ({mism} mismatches)')
    cl['Shipments'], cl['Delivery_Performance'] = sh, dp
    return cl

def build_order_master(cl):
    o, oi, sh, rt, c, w = cl['Orders'], cl['Order_Items'], cl['Shipments'], cl['Returns'], cl['Customers'], cl['Warehouses']
    g = oi.groupby('OrderID').agg(n_lines=('ProductID', 'size'), units=('Quantity', 'sum'), order_value=('LineValue', 'sum'), n_categories=('Category', 'nunique')).reset_index()
    dom = oi.groupby(['OrderID', 'Category']).LineValue.sum().reset_index().sort_values(['OrderID', 'LineValue'], ascending=[True, False]).drop_duplicates('OrderID')[['OrderID', 'Category']].rename(columns={'Category': 'DominantCategory'})
    r = rt.groupby('OrderID').agg(n_returns=('ReturnID', 'size'), ReturnReasons=('Reason', lambda s: ' | '.join(sorted(s)))).reset_index()
    rd = pd.crosstab(rt.OrderID, rt.Reason).add_prefix('ret_').reset_index()
    m = (o.merge(c[['CustomerID', 'City']].rename(columns={'City': 'CustomerCity'}), on='CustomerID', how='left').merge(g, on='OrderID', how='left').merge(dom, on='OrderID', how='left')
         .merge(sh[['OrderID', 'ShipmentID', 'WarehouseID', 'DelayDays', 'IsDelayed', 'DelaySeverity']], on='OrderID', how='left')
         .merge(w.rename(columns={'Location': 'WarehouseCity'}), on='WarehouseID', how='left').merge(r, on='OrderID', how='left').merge(rd, on='OrderID', how='left'))
    for col in ['n_lines', 'units', 'order_value', 'n_categories', 'n_returns'] + [x for x in m if x.startswith('ret_')]: m[col] = m[col].fillna(0)
    m['HasItems'] = (m.n_lines > 0).astype(int); m['HasReturn'] = (m.n_returns > 0).astype(int)
    m['DominantCategory'] = m.DominantCategory.fillna('No items'); m['ReturnReasons'] = m.ReturnReasons.fillna('')
    m['SameCityDispatch'] = (m.CustomerCity == m.WarehouseCity).astype(int)
    return m

cl = clean_all(raw)
master = build_order_master(cl)
log('Integrate datasets', 'Orders + Customers + Order_Items + Products + Shipments + Warehouses + Returns', 'built the order-level master table (one row per order)', len(master), 'Single analytical grain for EDA, tests and ML features')
print(f'Cleaned tables: {len(cl)} · master table: {master.shape[0]:,} rows × {master.shape[1]} columns')
R['n_orders'] = len(master); R['n_customers_total'] = len(cl['Customers']); R['n_customers_ordering'] = int(master.CustomerID.nunique())
master.head(3).T

# %%
# Outlier detection (Tukey 1.5×IQR). Policy: values are VALID business quantities, so they are flagged, never deleted.
oi_c, inv_c, sh_c = cl['Order_Items'], cl['Inventory'], cl['Shipments']
targets = {'Order_Items.Quantity (consolidated)': oi_c.Quantity, 'Products.UnitPrice': cl['Products'].UnitPrice, 'Inventory.Stock (consolidated)': inv_c.Stock,
           'Shipments.DelayDays': sh_c.DelayDays, 'Order_Items.LineValue': oi_c.LineValue, 'Orders.order_value (orders with items)': master.loc[master.HasItems == 1, 'order_value']}
rows = []
for n, s in targets.items():
    f, lo, hi = iqr_flag(s); rows.append(dict(variable=n, n=len(s), lower_fence=lo, upper_fence=hi, outliers=int(f.sum()), outlier_pct=100*f.mean(), max=s.max()))
outl = pd.DataFrame(rows); R['outliers'] = outl.set_index('variable').outliers.to_dict()
# Treatment: keep rows; add a capped copy of order_value (99th pct) for algorithms sensitive to extremes
cap = master.loc[master.HasItems == 1, 'order_value'].quantile(.99); master['order_value_capped'] = master.order_value.clip(upper=cap)
log('Detect & treat outliers', 'All numeric', 'IQR flags; no rows removed; order_value_capped (99th percentile cap) added for modelling', int(outl.outliers.sum()), 'Extreme-but-valid values carry business meaning (large baskets)')
print('Cap used for order_value_capped:', round(cap, 2)); outl

# %%
# Sensitivity of the inventory consolidation assumption: do warehouse stock rankings change if repeated keys are averaged instead of summed?
inv_raw = raw['Inventory'].drop_duplicates()
a = inv_raw.groupby('WarehouseID').Stock.sum(); b = inv_raw.groupby(['WarehouseID', 'ProductID']).Stock.mean().groupby('WarehouseID').sum()
rho = stats.spearmanr(a, b).statistic
R['inv_sensitivity_rho'] = float(rho)
print(f'Spearman rank correlation of warehouse stock ranking, "sum of lots" vs "average of lots": {rho:.3f}')
print('→ Warehouse conclusions are largely stable under the lot-consolidation assumption (rank correlation about 0.8, not identical).' if rho > .8 else '→ Warehouse stock conclusions are sensitive to this assumption; treat with caution.')

# %%
# Save cleaned datasets
out = {'Customers': cl['Customers'], 'Orders': cl['Orders'], 'Order_Items': cl['Order_Items'], 'Products': cl['Products'], 'Suppliers': cl['Suppliers'], 'Warehouses': cl['Warehouses'],
       'Inventory': cl['Inventory'], 'Shipments': cl['Shipments'], 'Delivery_Performance': cl['Delivery_Performance'], 'Employees': cl['Employees'], 'Returns': cl['Returns'], 'Orders_Master': master}
for k, d in out.items(): d.to_csv(CLEAN/f'SwiftLog_{k}_clean.csv', index=False)
print('Saved', len(out), 'cleaned datasets to', CLEAN.relative_to(BASE))
pd.DataFrame(LOG)

# %% [md]
# ### Automated data-profiling report
# `ydata-profiling` / Sweetviz could not be installed in the build environment (package index blocked), so this notebook generates an equivalent automated profile with pandas + matplotlib
# (per-column type, missingness, distinct count, distribution, descriptive statistics, alerts, correlations) → `Reports/Data_Profiling_Report.html`.
# If `ydata-profiling` is installed in your environment, the second half of the cell also produces the official report.

# %%
def _hist_b64(s, numeric=True):
    fig, ax = plt.subplots(figsize=(3.6, 1.5)); ax.grid(False)
    if numeric: ax.hist(s.dropna(), bins=min(30, max(5, s.nunique())), color=BLUE)
    else: vc = s.value_counts().head(8); ax.barh(vc.index.astype(str)[::-1], vc.values[::-1], color=BLUE)
    ax.tick_params(labelsize=6); [sp.set_visible(False) for sp in ax.spines.values()]
    b = io.BytesIO(); fig.savefig(b, format='png', dpi=80, bbox_inches='tight'); plt.close(fig); return base64.b64encode(b.getvalue()).decode()

def profile_html(tables, path, title):
    css = ('body{font-family:Arial,sans-serif;margin:24px;color:#0b0b0b;background:#fff}h1{color:#16345c}h2{background:#16345c;color:#fff;padding:8px 12px;border-radius:4px;margin-top:34px}'
           '.k{display:inline-block;margin:4px 10px 10px 0;padding:8px 14px;background:#eaf2fc;border-radius:6px}.card{display:inline-block;vertical-align:top;width:380px;margin:6px;padding:10px;border:1px solid #dcdcd8;border-radius:6px;font-size:12px}'
           'table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #dcdcd8;padding:3px 7px;text-align:right}th{background:#f1f1ee}.warn{color:#b3380d}')
    h = [f'<html><head><meta charset="utf-8"><title>{title}</title><style>{css}</style></head><body><h1>{title}</h1>',
         f'<p>Generated {pd.Timestamp.today().date()} by the AnalystLab notebook profiler (pandas + matplotlib). Tables profiled: {len(tables)}.</p>']
    h.append('<h2>Overview</h2><table><tr><th>Table</th><th>Rows</th><th>Cols</th><th>Missing cells</th><th>Duplicate rows</th><th>Memory (KB)</th></tr>')
    for n, d in tables.items(): h.append(f'<tr><td style="text-align:left">{n}</td><td>{len(d):,}</td><td>{d.shape[1]}</td><td>{int(d.isna().sum().sum())}</td><td>{int(d.duplicated().sum())}</td><td>{d.memory_usage(deep=True).sum()/1024:,.0f}</td></tr>')
    h.append('</table>')
    for n, d in tables.items():
        h.append(f'<h2>{n}</h2>'); alerts = []
        for c in d.columns:
            s = d[c]; num = pd.api.types.is_numeric_dtype(s) and s.dtype != bool; nun = s.nunique()
            if nun == len(s) and len(s) > 1: alerts.append(f'{c}: all values unique (identifier)')
            if nun == 1: alerts.append(f'{c}: constant column')
            if num and nun > 2 and abs(stats.skew(s.dropna())) > 1: alerts.append(f'{c}: highly skewed (skew={stats.skew(s.dropna()):.2f})')
            if s.isna().any(): alerts.append(f'{c}: {s.isna().mean():.1%} missing')
            if num and (s == 0).mean() > .3: alerts.append(f'{c}: {(s==0).mean():.0%} zeros')
        h.append('<p class="warn"><b>Alerts:</b> ' + ('; '.join(alerts) if alerts else 'none') + '</p>')
        for c in d.columns:
            s = d[c]; num = pd.api.types.is_numeric_dtype(s) and s.dtype != bool and s.nunique() > 1
            h.append(f'<div class="card"><b>{c}</b> <i>({s.dtype})</i><br>distinct {s.nunique():,} · missing {s.isna().sum()} ({s.isna().mean():.1%})<br>')
            if num: h.append(f'mean {s.mean():,.2f} · median {s.median():,.2f} · std {s.std():,.2f}<br>min {s.min():,.2f} · max {s.max():,.2f} · skew {stats.skew(s.dropna()):.2f} · kurt {stats.kurtosis(s.dropna()):.2f}<br>')
            elif s.nunique() <= 20: h.append('top: ' + ', '.join(f'{k} ({v})' for k, v in s.value_counts().head(5).items()) + '<br>')
            elif not pd.api.types.is_datetime64_any_dtype(s): h.append(f'e.g. {", ".join(map(str, s.head(3)))}<br>')
            if pd.api.types.is_datetime64_any_dtype(s): h.append(f'range {s.min().date()} → {s.max().date()}')
            elif num or 1 < s.nunique() <= 40: h.append(f'<img src="data:image/png;base64,{_hist_b64(s, num)}">')
            h.append('</div>')
    nums = master.select_dtypes('number').loc[:, lambda x: x.nunique() > 1]
    fig, ax = plt.subplots(figsize=(9, 7)); sns.heatmap(nums.corr(), cmap='RdBu_r', center=0, annot=False, ax=ax); ax.set_title('Correlation matrix — Orders_Master (numeric)')
    b = io.BytesIO(); fig.savefig(b, format='png', dpi=90, bbox_inches='tight'); plt.close(fig)
    h.append(f'<h2>Correlations</h2><img src="data:image/png;base64,{base64.b64encode(b.getvalue()).decode()}"></body></html>')
    Path(path).write_text('\n'.join(h), encoding='utf-8')

profile_html(out, REPORTS/'Data_Profiling_Report.html', 'SwiftLog — Automated Data Profiling Report')
print('Wrote', (REPORTS/'Data_Profiling_Report.html').relative_to(BASE), f'({(REPORTS/"Data_Profiling_Report.html").stat().st_size/1e6:.1f} MB)')
try:
    from ydata_profiling import ProfileReport
    ProfileReport(master.drop(columns=['OrderDate']), title='SwiftLog Orders Master', minimal=True).to_file(REPORTS/'Data_Profiling_Report_ydata.html'); print('ydata-profiling report also written.')
except ImportError:
    print('ydata-profiling not installed here — pandas profiler used (install with: pip install ydata-profiling).')
