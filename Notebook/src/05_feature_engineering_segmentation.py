# %% [md]
# ---
# # PART 5 — Feature Engineering
# **Grain:** one row per order (5,000). **Targets:** `IsDelayed` (delay ≥ 3 days), `DelayDays`, `HasReturn`.
# **Leakage rule:** any feature about the customer or warehouse uses only information available *before* the order (expanding, shifted history). Post-outcome fields (`DelayDays`, `n_returns`, `ReturnReasons`, `Status` for a not-yet-final order) are never used as predictors.
# Every feature is documented with: why it was created, its business value and its ML value (see the feature dictionary below).

# %%
fe = master.sort_values(['OrderDate', 'OrderID']).reset_index(drop=True).copy()
fe['OrderDate'] = pd.to_datetime(fe.OrderDate)
FD = []  # feature dictionary
def feat(name, why, biz, ml, group): FD.append(dict(feature=name, group=group, why_created=why, business_value=biz, ml_value=ml))

# --- Time features
fe['IsWeekend'] = (fe.OrderDate.dt.dayofweek >= 5).astype(int); fe['DayOfMonth'] = fe.OrderDate.dt.day; fe['IsMonthEnd'] = (fe.OrderDate.dt.day >= 25).astype(int)
fe['DaysSinceStart'] = (fe.OrderDate - fe.OrderDate.min()).dt.days
feat('IsWeekend', 'Weekend dispatch often has thinner staffing', 'Tests whether weekend orders need different cut-offs', 'Binary calendar signal; captures weekly seasonality', 'Time')
feat('IsMonthEnd', 'Month-end (day ≥ 25) volume peaks are common in logistics', 'Flags peak-load days for roster planning', 'Captures intra-month cycles', 'Time')
feat('DaysSinceStart', 'Continuous time index', 'Lets trend be tracked', 'Allows a model to learn drift; also used for the time-ordered train/test split', 'Time')

# --- Order / basket features
fe['value_per_unit'] = fe.order_value / fe.units.replace(0, np.nan); fe['log_order_value'] = np.log1p(fe.order_value)
fe['units_per_line'] = fe.units / fe.n_lines.replace(0, np.nan); fe['is_multi_category'] = (fe.n_categories > 1).astype(int)
q75 = fe.order_value.quantile(.75); fe['is_high_value'] = (fe.order_value >= q75).astype(int); fe['is_bulk'] = (fe.units >= fe.units.quantile(.75)).astype(int)
feat('value_per_unit', 'Average price of what is in the basket', 'Separates premium from commodity baskets', 'Scale-free value signal, less correlated with size than order_value', 'Basket')
feat('log_order_value', 'Order value is right-skewed (skew ≈ 1)', 'None directly (modelling aid)', 'Reduces skew for linear/distance-based models', 'Basket')
feat('units_per_line', 'Handling effort per product line', 'Proxy for pick complexity', 'Captures bulk vs spread baskets', 'Basket')
feat('is_multi_category', 'Multi-category orders may need multiple picks', 'Flags complex fulfilment', 'Binary complexity indicator', 'Basket')
feat('is_high_value / is_bulk', 'Top-quartile value / size', 'Identifies orders worth prioritising or protecting', 'Non-linear threshold features for tree and linear models', 'Basket')

# --- Customer history (strictly prior orders)
fe['cust_prior_orders'] = fe.groupby('CustomerID').cumcount()
fe['cust_prior_value'] = fe.groupby('CustomerID').order_value.cumsum() - fe.order_value
fe['cust_avg_prior_value'] = (fe.cust_prior_value / fe.cust_prior_orders.replace(0, np.nan))
fe['days_since_prev_order'] = fe.groupby('CustomerID').OrderDate.diff().dt.days
fe['is_repeat_customer'] = (fe.cust_prior_orders > 0).astype(int)
fe['value_vs_cust_avg'] = (fe.order_value / fe.cust_avg_prior_value.replace(0, np.nan))
feat('cust_prior_orders', 'Order count before this order', 'Loyalty/frequency, the F in RFM', 'Captures customer behaviour without leaking the current outcome', 'Customer')
feat('cust_prior_value', 'Cumulative spend before this order', 'Customer lifetime value to date (the M in RFM)', 'Monetary history signal', 'Customer')
feat('cust_avg_prior_value', 'Average prior basket', 'Typical spend level', 'Baseline for detecting unusual baskets', 'Customer')
feat('days_since_prev_order', 'Recency gap between orders', 'Early-warning for churn (long gaps)', 'Recency signal (the R in RFM)', 'Customer')
feat('is_repeat_customer', 'First order vs returning', 'Onboarding vs retention experience', 'Simple, stable binary', 'Customer')
feat('value_vs_cust_avg', 'This basket relative to customer norm', 'Spots unusually large/small orders', 'Normalised value; reduces between-customer variance', 'Customer')

# --- Warehouse features (static capacity proxies + prior performance)
whs = pd.DataFrame({'wh_staff': emp, 'wh_total_stock': wi.stock, 'wh_low_stock_share': wi.low / wi.positions, 'wh_shipments_total': wl.shipments}); whs['wh_workload_per_emp'] = whs.wh_shipments_total / whs.wh_staff
fe = fe.merge(whs[['wh_staff', 'wh_total_stock', 'wh_low_stock_share', 'wh_workload_per_emp']], left_on='WarehouseID', right_index=True, how='left')
fe['wh_prior_delay_rate'] = fe.groupby('WarehouseID').IsDelayed.transform(lambda s: s.shift().rolling(50, min_periods=10).mean())
fe['wh_prior_avg_delay'] = fe.groupby('WarehouseID').DelayDays.transform(lambda s: s.shift().rolling(50, min_periods=10).mean())
_tmp = []
for w, g in fe.groupby('WarehouseID'):
    s = g.set_index('OrderDate').OrderID.rolling('7D').count() - 1; _tmp.append(pd.Series(s.values, index=g.index))
fe['wh_orders_last7d'] = pd.concat(_tmp).sort_index()
fe['wh_load_index'] = fe.wh_orders_last7d / fe.wh_staff
feat('wh_staff', 'Headcount at dispatch site', 'Capacity proxy (true capacity not in data)', 'Static site attribute', 'Warehouse')
feat('wh_total_stock / wh_low_stock_share', 'Site stock level and share of positions below 20 units', 'Replenishment risk indicator', 'Stock-pressure signal at the site', 'Warehouse')
feat('wh_workload_per_emp', 'Season-long shipments ÷ staff', 'Staffing imbalance', 'Site load level', 'Warehouse')
feat('wh_prior_delay_rate / wh_prior_avg_delay', 'Rolling delay over the previous 50 shipments of the same warehouse', 'Live reliability monitor', 'Time-safe warehouse performance feature (no look-ahead)', 'Warehouse')
feat('wh_orders_last7d / wh_load_index', 'Orders handled by the site in the previous 7 days, per employee', 'Short-term congestion', 'Dynamic load feature', 'Warehouse')

# --- Route / geography
fe['route'] = fe.WarehouseCity + '→' + fe.CustomerCity
feat('SameCityDispatch', 'Warehouse and customer in the same city', 'Local vs long-haul fulfilment', 'Distance proxy (no coordinates in data)', 'Route')
feat('route', 'Origin city → destination city', 'Lane-level reporting', 'Categorical lane feature (12 combos)', 'Route')

# --- Encodings
cat_cols = ['CustomerCity', 'WarehouseCity', 'DominantCategory']
oh = pd.get_dummies(fe[cat_cols], prefix=['cust_city', 'wh_city', 'cat'], dtype=int); fe = pd.concat([fe, oh], axis=1)
fe['Status_code'] = fe.Status.map({'Cancelled': 0, 'Pending': 1, 'Delivered': 2}); fe['DayOfWeek_num'] = fe.OrderDate.dt.dayofweek; fe['Month_sin'] = np.sin(2*np.pi*fe.Month/12); fe['Month_cos'] = np.cos(2*np.pi*fe.Month/12)
feat('one-hot: cust_city_*, wh_city_*, cat_*', 'Nominal variables cannot be used raw by most models', 'None directly', 'One-hot encoding avoids false ordering (few levels, so no high-cardinality issue)', 'Encoding')
feat('Month_sin / Month_cos', 'Cyclical encoding of month', 'None directly', 'Lets models see December and January as neighbours', 'Encoding')
feat('Status_code', 'Ordinal status (Cancelled < Pending < Delivered)', 'Funnel stage', 'Ordinal encoding; NOT used as a delay predictor because status can be set after the outcome', 'Encoding')

# --- Return-risk flags (targets & diagnostics)
fe['target_is_delayed'] = fe.IsDelayed; fe['target_delay_days'] = fe.DelayDays; fe['target_has_return'] = fe.HasReturn
fe = fe.replace([np.inf, -np.inf], np.nan)
FD_df = pd.DataFrame(FD); R['n_features_documented'] = len(FD_df)
print(f'Feature-engineered table: {fe.shape[0]:,} rows × {fe.shape[1]} columns. {len(FD_df)} feature groups documented.')
FD_df

# %%
# Missing values are structural (first orders have no history) - impute with explicit flags rather than silently filling
hist_cols = ['cust_avg_prior_value', 'days_since_prev_order', 'value_vs_cust_avg', 'wh_prior_delay_rate', 'wh_prior_avg_delay', 'value_per_unit', 'units_per_line']
miss = fe[hist_cols].isna().sum().to_frame('missing'); miss['why'] = ['first order of customer', 'first order of customer', 'first order of customer', 'warehouse warm-up (<10 shipments)', 'warehouse warm-up (<10 shipments)', 'order has no line items', 'order has no line items']
R['fe_missing'] = miss.missing.to_dict()
miss

# %%
# Final modelling table: impute (median) + keep raw columns for transparency; save
model_df = fe.copy()
for c in hist_cols: model_df[c + '_missing'] = model_df[c].isna().astype(int); model_df[c] = model_df[c].fillna(model_df[c].median())
model_df.to_csv(f'{BASE}/Data/Feature_Engineered_Dataset.csv', index=False)
R['fe_shape'] = list(model_df.shape)
print(f'Saved Data/Feature_Engineered_Dataset.csv {model_df.shape}')

# %%
# Feature screening against targets (non-parametric): which engineered features carry ANY signal?
cand = ['units', 'order_value', 'n_lines', 'n_categories', 'value_per_unit', 'units_per_line', 'cust_prior_orders', 'cust_prior_value', 'days_since_prev_order', 'value_vs_cust_avg', 'wh_staff', 'wh_total_stock', 'wh_low_stock_share', 'wh_workload_per_emp', 'wh_prior_delay_rate', 'wh_orders_last7d', 'wh_load_index', 'SameCityDispatch', 'IsWeekend', 'IsMonthEnd', 'DaysSinceStart', 'is_repeat_customer', 'is_high_value', 'is_multi_category']
sc = pd.DataFrame({c: dict(rho_delay=stats.spearmanr(model_df[c], model_df.DelayDays)[0], p_delay=stats.spearmanr(model_df[c], model_df.DelayDays)[1], rho_return=stats.spearmanr(model_df[c], model_df.HasReturn)[0], p_return=stats.spearmanr(model_df[c], model_df.HasReturn)[1]) for c in cand}).T
sc['abs_max'] = sc[['rho_delay', 'rho_return']].abs().max(axis=1); sc = sc.sort_values('abs_max', ascending=False)
R['feature_screen_max'] = float(sc.abs_max.max()); R['feature_screen_top'] = sc.head(5).round(4).to_dict('index')
n_sig = int(((sc.p_delay < .05) | (sc.p_return < .05)).sum()); R['feature_screen_nsig'] = n_sig; print(f'{n_sig} of {len(sc)} features reach p<0.05 against a target (about {len(sc)*2*.05:.0f} expected by chance alone); strongest |rho| = {sc.abs_max.max():.3f}.')
sc.round(4).head(10)

# %%
# Baseline predictive test (time-ordered split: first 70% train, last 30% test) – what can ML achieve today?
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
X_cols = [c for c in cand] + [c for c in model_df.columns if c.startswith(('cust_city_', 'wh_city_', 'cat_'))] + ['Month_sin', 'Month_cos', 'DayOfWeek_num']
cut = int(len(model_df) * .7); tr, te = model_df.iloc[:cut], model_df.iloc[cut:]
res = []
for tgt in ['IsDelayed', 'HasReturn']:
    base = max(te[tgt].mean(), 1 - te[tgt].mean())
    for name, m in [('Logistic regression', make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))), ('Random forest', RandomForestClassifier(300, min_samples_leaf=20, random_state=7, n_jobs=-1)), ('Gradient boosting', GradientBoostingClassifier(random_state=7, max_depth=2))]:
        m.fit(tr[X_cols], tr[tgt]); pr = m.predict_proba(te[X_cols])[:, 1]; res.append(dict(target=tgt, model=name, test_AUC=roc_auc_score(te[tgt], pr), accuracy=float(((pr > .5) == te[tgt]).mean()), majority_baseline=base))
ML = pd.DataFrame(res); R['ml_baseline'] = ML.round(4).to_dict('records'); R['best_auc_delay'] = float(ML[ML.target == 'IsDelayed'].test_AUC.max()); R['best_auc_return'] = float(ML[ML.target == 'HasReturn'].test_AUC.max())
print('AUC 0.50 = coin flip. Anything under ~0.60 means the features carry essentially no usable signal for that target.')
ML.round(3)

# %%
rf = RandomForestClassifier(300, min_samples_leaf=20, random_state=7, n_jobs=-1).fit(tr[X_cols], tr.IsDelayed)
imp = pd.Series(rf.feature_importances_, X_cols).sort_values(ascending=False).head(12)[::-1]
fig, ax = plt.subplots(figsize=(8, 4.8)); ax.barh(imp.index, imp.values, color=BLUE, height=.6); ax.set_title('Random-forest feature importance for predicting delay'); ax.set_xlabel('Importance (impurity-based)'); ax.set_ylabel('Engineered feature')
chart('24_feature_importance', 'Feature importance (delay model)', f"Importance is spread thinly across features, led by {imp.index[-1]} and {imp.index[-2]}; with a test AUC of only {R['best_auc_delay']:.2f} the ranking mostly reflects noise rather than real drivers.",
      'Today\'s extract cannot predict delays. Useful ML needs new inputs: carrier, route distance, promised vs actual dates, picker/shift, weather and stock availability at pick time.')

# %% [md]
# ---
# # PART 6 — Segmentation & Pattern Discovery
# We segment **customers** (RFM + K-Means), **products** (ABC revenue class × return/velocity) and **warehouses** (performance clustering as the supplier substitute). Suppliers cannot be segmented: no supplier key exists on any transactional table.

# %%
import textwrap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
asof = fe.OrderDate.max() + pd.Timedelta(days=1); cf = fe[fe.Status != 'Cancelled']
rfm = cf.groupby('CustomerID').agg(recency=('OrderDate', lambda s: (asof - s.max()).days), frequency=('OrderID', 'size'), monetary=('order_value', 'sum'), avg_basket=('order_value', 'mean'), delay_rate=('IsDelayed', 'mean'), return_rate=('HasReturn', 'mean'))
nov = rfm.monetary < 500  # customers whose confirmed orders carry no line items (order-capture gap) - kept out of clustering
R['rfm_no_value_customers'] = int(nov.sum()); rfm_all = rfm.copy(); rfm = rfm[~nov].copy()
print(f'{int(nov.sum())} customers have confirmed orders but (almost) no line-item value; they are reported separately, not clustered.')
Z = StandardScaler().fit_transform(np.log1p(rfm[['recency', 'frequency', 'monetary']]))
sil = {k: silhouette_score(Z, KMeans(k, n_init=10, random_state=7).fit_predict(Z)) for k in range(2, 8)}; R['silhouette'] = {int(k): float(v) for k, v in sil.items()}
fig, ax = plt.subplots(figsize=(6.5, 3.6)); ax.plot(list(sil), list(sil.values()), color=BLUE, marker='o', lw=2.2); ax.set_title('Choosing k for customer clusters'); ax.set_xlabel('Number of clusters (k)'); ax.set_ylabel('Silhouette score'); ax.set_ylim(0, .5)
chart('25_kmeans_silhouette', 'Cluster count selection', f"Silhouette scores are modest (best {max(sil.values()):.2f} at k = {max(sil, key=sil.get)}); anything below ~0.5 signals clusters that overlap rather than form distinct groups.",
      'Customer segments here are useful as planning buckets (high/low value) but are not natural, sharply separated groups — avoid over-interpreting them.')

# %%
K = 4; rfm['cluster'] = KMeans(K, n_init=10, random_state=7).fit_predict(Z)
prof = rfm.groupby('cluster').agg(customers=('recency', 'size'), recency=('recency', 'mean'), frequency=('frequency', 'mean'), monetary=('monetary', 'mean'), avg_basket=('avg_basket', 'mean'), delay_rate=('delay_rate', 'mean'), return_rate=('return_rate', 'mean'))
def label(r):
    hv = r.monetary >= rfm.monetary.median(); rec = r.recency <= rfm.recency.median()
    return {(True, True): 'Champions (high value, recent)', (True, False): 'At-risk high value (lapsing)', (False, True): 'Active low value', (False, False): 'Dormant low value'}[(hv, rec)]
prof['segment'] = prof.apply(label, axis=1)
for sname in prof.segment[prof.segment.duplicated(keep=False)].unique():  # disambiguate clusters that fall in the same quadrant
    idx = prof.index[prof.segment == sname]; keep = prof.loc[idx, 'monetary'].idxmax()
    for i in idx:
        if i != keep: prof.loc[i, 'segment'] = 'Newly active (recent, mid spend)'
names = prof.segment.to_dict(); rfm['segment'] = rfm.cluster.map(names)
prof['share_of_confirmed_revenue'] = rfm.groupby('cluster').monetary.sum() / rfm_all.monetary.sum(); R['segments'] = prof.round(3).to_dict('index')
print(f'Segment labels are rule-based on cluster means vs the median cluster; {rfm.shape[0]:,} customers with ≥1 confirmed order.')
prof.round(2)

# %%
seg = prof.sort_values('monetary', ascending=False); seg['lab'] = [textwrap.fill(x, 16) for x in seg.segment]; fig, axs = plt.subplots(1, 2, figsize=(11.5, 4))
axs[0].bar(seg.lab, seg.customers, color=BLUE, width=.55); axs[0].set_title('Customers per segment'); axs[0].set_xlabel('Segment'); axs[0].set_ylabel('Customers'); axs[0].tick_params(axis='x', labelsize=8.5)
for i, v in enumerate(seg.customers): axs[0].text(i, v + 4, int(v), ha='center', fontsize=9)
axs[1].bar(seg.lab, seg.share_of_confirmed_revenue*100, color=AQUA, width=.55); axs[1].set_title('Share of confirmed revenue'); axs[1].set_xlabel('Segment'); axs[1].set_ylabel('% of confirmed revenue'); axs[1].tick_params(axis='x', labelsize=8.5)
for i, v in enumerate(seg.share_of_confirmed_revenue*100): axs[1].text(i, v + .6, f'{v:.0f}%', ha='center', fontsize=9)
top_seg = seg.iloc[0]
chart('26_customer_segments', 'Customer segments (RFM + K-Means)', f"{top_seg.segment.split(' (')[0]} ({int(top_seg.customers)} customers) average {short(top_seg.monetary)} each and hold {pct(top_seg.share_of_confirmed_revenue,0)} of confirmed revenue. Segments differ in recency and frequency but all share the same delay rate (≈{pct(prof.delay_rate.mean(),0)}).",
      'Retention effort should go to the high-value customers whose last order is old (lapsing) before they are lost; every segment suffers the same delivery experience, so service reliability is a retention risk for all of them.')

# %%
pa = oi_conf.groupby('ProductID').agg(revenue=('LineValue', 'sum'), units=('Quantity', 'sum'), orders=('OrderID', 'nunique')).join(cl['Products'].set_index('ProductID')[['Category', 'UnitPrice']])
rt_p = cl['Returns'].merge(cl['Order_Items'][['OrderID', 'ProductID']], on='OrderID').groupby('ProductID').size().rename('returns_touching'); pa = pa.join(rt_p).fillna({'returns_touching': 0})
pa['rev_rank_share'] = pa.revenue.rank(ascending=False) / len(pa); pa['ABC'] = np.where(pa.rev_rank_share <= .2, 'A (top 20%)', np.where(pa.rev_rank_share <= .5, 'B (next 30%)', 'C (bottom 50%)'))
abc = pa.groupby('ABC').agg(products=('revenue', 'size'), revenue=('revenue', 'sum'), units=('units', 'sum')); abc['revenue_share'] = abc.revenue / abc.revenue.sum(); R['abc'] = abc.round(3).to_dict('index')
xt = pd.crosstab(pa.ABC, pa.Category)
fig, ax = plt.subplots(figsize=(8.5, 4)); bottom = np.zeros(3); catcol = {'Electronics': BLUE, 'Fashion': ORANGE, 'Food': AQUA}
for ccat in ['Electronics', 'Fashion', 'Food']:
    ax.bar(xt.index, xt[ccat], bottom=bottom, color=catcol[ccat], label=ccat, width=.55, edgecolor='white', linewidth=1.5); bottom += xt[ccat].values
ax.set_title('ABC product classes by category'); ax.set_xlabel('Revenue class'); ax.set_ylabel('Products'); ax.legend(ncol=3, loc='upper left')
chart('27_abc_classes', 'Product ABC segmentation', f"Class A (top 20%, {int(abc.loc['A (top 20%)','products'])} products) earns {pct(abc.loc['A (top 20%)','revenue_share'])} of confirmed revenue; class C (bottom 50%) still earns {pct(abc.loc['C (bottom 50%)','revenue_share'])}. All three categories appear in every class.",
      'ABC differences are mild, so a tail clean-up would give up real revenue. Manage class A for availability (never stock out), and review class C on margin and return data before deleting any line.')

# %%
wk = wl[['ship_per_emp']].join(dw[['rate', 'md']]).join(wi[['low', 'stock']]); wk = wk.join(pd.DataFrame({'return_rate': master.groupby('WarehouseID').HasReturn.mean()}))
wk.columns = ['workload_per_emp', 'delay_rate', 'mean_delay', 'low_stock_positions', 'stock', 'return_rate']
Zw = StandardScaler().fit_transform(wk[['workload_per_emp', 'delay_rate', 'low_stock_positions', 'return_rate']]); wk['cluster'] = KMeans(3, n_init=20, random_state=7).fit_predict(Zw)
sw = silhouette_score(Zw, wk.cluster); R['wh_silhouette'] = float(sw)
# reliability score: z-score composite (lower delay, lower return, fewer low-stock = better)
zs = lambda s: (s - s.mean()) / s.std(); wk['reliability_score'] = -(zs(wk.delay_rate) + zs(wk.return_rate) + zs(wk.low_stock_positions)) / 3
wk['rank'] = wk.reliability_score.rank(ascending=False).astype(int); R['wh_reliability'] = wk.round(3).to_dict('index')
fig, ax = plt.subplots(figsize=(8.5, 5)); cm_ = {0: BLUE, 1: ORANGE, 2: AQUA}
for cl_, d in wk.groupby('cluster'): ax.scatter(d.workload_per_emp, d.delay_rate*100, s=80, color=cm_[cl_], label=f'Cluster {cl_ + 1} ({len(d)} sites)', edgecolor='white', lw=1.2)
for w, r in wk.iterrows(): ax.annotate(w, (r.workload_per_emp, r.delay_rate*100), xytext=(6, 4), textcoords='offset points', fontsize=8.5)
ax.set_title('Warehouse clusters on workload, delay, stock risk and returns'); ax.set_xlabel('Shipments per employee'); ax.set_ylabel('% shipments delayed'); ax.legend(loc='best')
chart('28_warehouse_clusters', 'Warehouse performance clusters', f"Three warehouse clusters emerge (silhouette {sw:.2f}). Best composite reliability: {wk.reliability_score.idxmax()}; weakest: {wk.reliability_score.idxmin()}. Because delay differences are statistically insignificant, clusters are driven mostly by workload and stock risk.",
      'Use the composite Reliability Score as a monitoring dashboard, not a ranking for blame. It is the best supplier-performance substitute available until supplier IDs are captured on inventory and shipments.')

# %%
dom = master.groupby('DominantCategory').agg(orders=('OrderID', 'size'), delay=('IsDelayed', 'mean'), ret=('HasReturn', 'mean'), cancel=('Status', lambda s: (s == 'Cancelled').mean()), aov=('order_value', 'mean'))
dow = master.groupby('DayOfWeek').agg(orders=('OrderID', 'size'), delay=('IsDelayed', 'mean')); R['dow'] = dow.round(3).to_dict('index'); R['dom_cat'] = dom.round(3).to_dict('index')
order_d = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']; dow = dow.reindex([d for d in order_d if d in dow.index])
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4)); axs[0].bar(dow.index, dow.orders, color=BLUE, width=.6); axs[0].set_title('Orders by weekday'); axs[0].set_xlabel('Weekday'); axs[0].set_ylabel('Orders'); axs[0].tick_params(axis='x', labelrotation=30)
axs[1].bar(dow.index, dow.delay*100, color=ORANGE, width=.6); axs[1].set_title('% delayed by weekday'); axs[1].set_xlabel('Weekday'); axs[1].set_ylabel('% shipments delayed'); axs[1].set_ylim(0, 80); axs[1].tick_params(axis='x', labelrotation=30)
chart('29_weekday_patterns', 'Weekday patterns', f"Order volume is flat across weekdays ({dow.orders.min()}–{dow.orders.max()} orders) and delay rates stay within {pct(dow.delay.min(),0)}–{pct(dow.delay.max(),0)}; no weekday effect.",
      'There is no peak day to staff for, and no day where delays are worse; weekly roster changes would not help. Orders also arrive every day of the week including weekends, which is unusual and worth confirming with the order system owners.')
print(f'Total charts produced: {len(CH)}')
