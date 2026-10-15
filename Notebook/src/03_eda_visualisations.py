# %% [md]
# ---
# # PART 3 — Exploratory Data Analysis
# **Revenue definition.** The raw data has no revenue field and order status is mixed (Delivered / Pending / Cancelled). We therefore report three explicit measures:
# *gross value* (all orders), **confirmed value** (all non-cancelled orders — used as the default "revenue" below) and *realised value* (Delivered only).
# Unless stated otherwise, currency units are as supplied (the data dictionary does not name the currency).

# %%
conf = master[master.Status != 'Cancelled'].copy()
oi_c = cl['Order_Items'].merge(master[['OrderID', 'Status']], on='OrderID')
oi_conf = oi_c[oi_c.Status != 'Cancelled']
emp = cl['Employees'].groupby('WarehouseID').size().rename('employees')
rev = pd.DataFrame({'Gross (all orders)': [master.order_value.sum()], 'Confirmed (excl. Cancelled)': [conf.order_value.sum()], 'Realised (Delivered only)': [master.loc[master.Status == 'Delivered', 'order_value'].sum()]}).T
rev.columns = ['value']; rev['share_of_gross'] = rev.value / rev.value.iloc[0]
R['rev_gross'], R['rev_confirmed'], R['rev_realised'] = [float(v) for v in rev.value]
R['cancel_rate'] = float((master.Status == 'Cancelled').mean()); R['pending_rate'] = float((master.Status == 'Pending').mean()); R['delivered_rate'] = float((master.Status == 'Delivered').mean())
R['cancelled_value'] = float(master.loc[master.Status == 'Cancelled', 'order_value'].sum())
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d; return c-h, c+h
R['delay_rate'] = float(master.IsDelayed.mean()); R['mean_delay'] = float(master.DelayDays.mean()); R['return_rate'] = float(master.HasReturn.mean()); R['n_returns'] = int(len(cl['Returns']))
# sensitivity: does the revenue definition change the product ranking?
def top(df, n=10): return set(df.groupby('ProductID').LineValue.sum().nlargest(n).index)
tg, tc, tr = top(oi_c), top(oi_conf), top(oi_c[oi_c.Status == 'Delivered'])
R['top10_overlap_conf_vs_gross'] = len(tg & tc); R['top10_overlap_conf_vs_delivered'] = len(tr & tc)
print(f"Cancelled orders = {pct(R['cancel_rate'])} of all orders and {short(R['cancelled_value'])} of gross value ({pct(R['cancelled_value']/R['rev_gross'])}).")
print(f"Top-10 product overlap — confirmed vs gross: {R['top10_overlap_conf_vs_gross']}/10 ; confirmed vs delivered-only: {R['top10_overlap_conf_vs_delivered']}/10  → product rankings depend materially on the status filter, so product conclusions carry that caveat.")
rev

# %% [md]
# ## 3.1 Univariate analysis — Customers, Orders, Products, Inventory, Shipments, Returns, Suppliers, Warehouses

# %%
cc = cl['Customers'].City.value_counts()
fig, ax = plt.subplots(figsize=(7, 3.6)); ax.bar(cc.index, cc.values, color=BLUE, width=.6)
for i, v in enumerate(cc.values): ax.text(i, v + 5, f'{v} ({pct(v/cc.sum(),0)})', ha='center', fontsize=9)
ax.set_title('Customers by city (n = 1,000)'); ax.set_ylabel('Number of customers'); ax.set_xlabel('Customer city'); ax.set_ylim(0, cc.max()*1.15)
chart('02_customers_by_city', 'Customers by city', f"Customers split almost evenly across four cities ({cc.index[0]} largest at {pct(cc.iloc[0]/cc.sum())}, {cc.index[-1]} smallest at {pct(cc.iloc[-1]/cc.sum())}).",
      'Demand is geographically balanced, so no single city dominates planning — but Ibadan holds the largest customer base yet has no SwiftLog warehouse (all Ibadan orders are dispatched from another city).')

# %%
ms = master.groupby(['YearMonth', 'Status']).size().unstack(fill_value=0)[['Delivered', 'Pending', 'Cancelled']]
fig, ax = plt.subplots(figsize=(9, 4.2)); bottom = np.zeros(len(ms)); cols = {'Delivered': BLUE, 'Pending': ORANGE, 'Cancelled': GREY}
for s in ms.columns:
    bars = ax.bar(ms.index, ms[s], bottom=bottom, color=cols[s], label=s, width=.65, edgecolor='white', linewidth=1.5); bottom += ms[s].values
bars = ax.patches[-1]; ax.text(len(ms)-1, bottom[-1] + 12, '154 orders\n(partial month,\ndata ends 8 Sep)', ha='center', fontsize=8, color='#52514e')
for i, v in enumerate(bottom[:-1]): ax.text(i, v + 12, f'{int(v)}', ha='center', fontsize=9)
ax.set_title('Monthly orders by status'); ax.set_ylabel('Orders'); ax.set_xlabel('Order month (2026)'); ax.legend(ncol=3, loc='upper left', bbox_to_anchor=(0, 1.02)); ax.set_ylim(0, 780)
full = ms.iloc[:-1].sum(axis=1); R['peak_month'] = full.idxmax(); R['peak_month_orders'] = int(full.max()); R['low_month'] = full.idxmin(); R['low_month_orders'] = int(full.min()); R['monthly_orders'] = full.to_dict()
chart('03_monthly_orders_by_status', 'Monthly orders by status', f"Order volume is steady at roughly {full.mean():.0f} orders a month (range {full.min()}–{full.max()}; peak {R['peak_month']}, low {R['low_month']}). In every month about a third of orders are Cancelled and a third are still Pending.",
      f"Only {pct(R['delivered_rate'])} of orders are Delivered while {pct(R['cancel_rate'])} are Cancelled — a persistent conversion leak worth {short(R['cancelled_value'])} in order value. Fixing cancellations is a bigger lever than adding demand.")

# %%
fig, ax = plt.subplots(figsize=(8, 4)); order = ['Electronics', 'Fashion', 'Food']; data = [cl['Products'].loc[cl['Products'].Category == c, 'UnitPrice'] for c in order]
bp = ax.boxplot(data, tick_labels=[f'{c}\n(n={len(d)})' for c, d in zip(order, data)], patch_artist=True, widths=.5, medianprops=dict(color=INK, lw=2))
for p in bp['boxes']: p.set(facecolor='#cfe0f7', edgecolor=BLUE)
ax.set_title('Product unit price by category'); ax.set_ylabel('Unit price (currency units)'); ax.set_xlabel('Product category')
pm = cl['Products'].groupby('Category').UnitPrice.agg(['mean', 'median', 'std']); R['price_by_cat'] = pm.round(2).to_dict()
chart('04_price_by_category', 'Unit price by category', f"All three categories span the full price range (≈{cl['Products'].UnitPrice.min():.0f}–{cl['Products'].UnitPrice.max():.0f}) with similar medians ({pm['median'].min():.0f}–{pm['median'].max():.0f}); category is not a proxy for price tier.",
      'Pricing is not differentiated by category, so category-level revenue differences come from volume and mix, not premium pricing — a pricing-strategy opportunity (premium vs value tiers).')

# %%
fig, ax = plt.subplots(figsize=(8.5, 4)); ax.hist(inv_c.Stock, bins=50, color=BLUE, edgecolor='white'); ax.axvline(LOW_STOCK, color=ORANGE, lw=2, ls='--')
ax.text(LOW_STOCK + 15, ax.get_ylim()[1]*.9, f'Low-stock line ({LOW_STOCK} units)\n{int(inv_c.LowStock.sum())} positions below', color=ORANGE, fontsize=9)
ax.set_title('Stock per warehouse–product position'); ax.set_xlabel('Units in stock (after consolidating stock lots)'); ax.set_ylabel('Positions')
R['low_stock_positions'] = int(inv_c.LowStock.sum()); R['zero_stock_positions'] = int((inv_c.Stock == 0).sum()); R['inv_positions'] = len(inv_c); R['total_stock'] = int(inv_c.Stock.sum())
chart('05_stock_distribution', 'Stock distribution', f"Stock per position is spread broadly (median {inv_c.Stock.median():.0f} units); only {R['low_stock_positions']} of {R['inv_positions']:,} positions ({pct(R['low_stock_positions']/R['inv_positions'])}) fall below {LOW_STOCK} units, {R['zero_stock_positions']} are at zero.",
      'Stock-outs are rare overall but not absent: the few low/zero positions are concentrated and can be fixed with targeted replenishment rather than a blanket stock increase. The long right tail comes from consolidated stock lots.')

# %%
dc = master.DelayDays.value_counts().sort_index()
fig, ax = plt.subplots(figsize=(8.5, 4)); ax.bar(dc.index, dc.values, color=[BLUE if d <= 2 else ORANGE for d in dc.index], width=.7)
for d, v in dc.items(): ax.text(d, v + 8, f'{v}', ha='center', fontsize=9)
ax.set_title('Shipment delay (days beyond promise)'); ax.set_xlabel('Delay days (0–2 = On Time, 3–7 = Delayed)'); ax.set_ylabel('Shipments'); ax.set_ylim(0, dc.max()*1.15)
ax.legend(handles=[mpatches.Patch(color=BLUE, label=f'On Time ({pct(1-R["delay_rate"])})'), mpatches.Patch(color=ORANGE, label=f'Delayed ({pct(R["delay_rate"])})')], loc='upper left')
chart('06_delay_distribution', 'Distribution of shipment delay', f"Delays are spread almost uniformly from 0 to 7 days (no peak at zero). {pct(R['delay_rate'])} of shipments are 'Delayed' (≥3 days); the mean delay is {R['mean_delay']:.2f} days.",
      'A uniform spread means lateness is systemic, not the result of a few bad events: fewer than 4 in 10 shipments are on time. Delivery reliability needs a network-wide process fix, not firefighting.')

# %%
rc = cl['Returns'].Reason.value_counts()
fig, ax = plt.subplots(figsize=(7.5, 3.8)); ax.barh(rc.index[::-1], rc.values[::-1], color=BLUE, height=.55)
for i, v in enumerate(rc.values[::-1]): ax.text(v + 3, i, f'{v} ({pct(v/rc.sum(),0)})', va='center', fontsize=9)
ax.set_title(f'Return reasons ({len(cl["Returns"])} return records)'); ax.set_xlabel('Return records'); ax.set_ylabel('Reason'); ax.set_xlim(0, rc.max()*1.2)
R['return_reasons'] = rc.to_dict()
chart('07_return_reasons', 'Return reasons', f"{rc.index[0]} is the most frequent reason ({pct(rc.iloc[0]/rc.sum())}), followed by {rc.index[1]} ({pct(rc.iloc[1]/rc.sum())}) and {rc.index[2]} ({pct(rc.iloc[2]/rc.sum())}); {pct(R['return_rate'])} of orders have at least one return.",
      'Roughly two-thirds of returns are controllable by operations (wrong item = picking accuracy; damaged = handling/packaging) rather than customer preference, so returns can be cut through warehouse process quality.')

# %%
wl = pd.DataFrame({'shipments': master.groupby('WarehouseID').size(), 'city': cl['Warehouses'].set_index('WarehouseID').Location}).join(emp); wl['ship_per_emp'] = wl.shipments / wl.employees
R['wh_load'] = wl.round(2).to_dict('index'); R['load_ratio'] = float(wl.ship_per_emp.max() / wl.ship_per_emp.min())
fig, ax = plt.subplots(figsize=(10, 4.2)); colr = [ORANGE if v > wl.ship_per_emp.mean()*1.25 else BLUE for v in wl.ship_per_emp]; ax.bar(wl.index, wl.ship_per_emp, color=colr, width=.65)
ax.axhline(wl.ship_per_emp.mean(), color=INK, ls='--', lw=1); ax.text(14.45, wl.ship_per_emp.mean() + .6, f'network mean {wl.ship_per_emp.mean():.1f}', ha='right', fontsize=9)
for i, (w, r) in enumerate(wl.iterrows()): ax.text(i, r.ship_per_emp + .5, f'{r.ship_per_emp:.0f}', ha='center', fontsize=8.5); ax.text(i, 1, f'{int(r.employees)} staff', ha='center', fontsize=7.5, color='white', rotation=90, va='bottom')
ax.set_title('Workload per employee by warehouse (shipments ÷ staff)'); ax.set_xlabel('Warehouse'); ax.set_ylabel('Shipments per employee (Jan–8 Sep)')
hi = wl.sort_values('ship_per_emp').index[-3:][::-1].tolist(); lo = wl.ship_per_emp.idxmin()
chart('08_workload_per_employee', 'Warehouse workload per employee', f"Shipments handled per employee range from {wl.ship_per_emp.min():.0f} ({lo}) to {wl.ship_per_emp.max():.0f} ({wl.ship_per_emp.idxmax()}) — a {R['load_ratio']:.1f}× spread. Heaviest loads: {', '.join(hi)}.",
      'Staffing is not aligned with workload: shipment volume is nearly equal across warehouses (≈280–370) but headcount varies 9–17. Rebalancing staff toward the overloaded sites is a low-cost capacity gain. (Warehouse capacity is not in the data, so this is a load proxy, not utilisation.)')

# %%
ov = master.loc[master.HasItems == 1, 'order_value']
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8)); axs[0].hist(ov, bins=40, color=BLUE, edgecolor='white'); axs[0].axvline(ov.mean(), color=ORANGE, lw=2); axs[0].axvline(ov.median(), color=INK, lw=1.5, ls='--')
axs[0].set_title('Order value distribution'); axs[0].set_xlabel('Order value (currency units)'); axs[0].set_ylabel('Orders'); axs[0].legend(['mean ' + short(ov.mean()), 'median ' + short(ov.median())], loc='upper right')
axs[1].hist(master.loc[master.HasItems == 1, 'units'], bins=30, color=AQUA, edgecolor='white'); axs[1].set_title('Units per order'); axs[1].set_xlabel('Units in order'); axs[1].set_ylabel('Orders')
R['aov_all'] = float(ov.mean()); R['aov_conf'] = float(conf.loc[conf.HasItems == 1, 'order_value'].mean()); R['orders_no_items'] = int((master.HasItems == 0).sum())
chart('09_order_value_distribution', 'Order value and basket size', f"Order value is right-skewed (mean {short(ov.mean())} > median {short(ov.median())}; skew {stats.skew(ov):.2f}); a typical basket is {master.loc[master.HasItems==1,'units'].median():.0f} units across {master.loc[master.HasItems==1,'n_lines'].median():.0f} products.",
      f"Baskets are large for a logistics catalogue, and {R['orders_no_items']} orders ({pct(R['orders_no_items']/len(master))}) carry no line items at all — an order-capture defect that hides revenue and should be closed at source.")

# %%
sup = cl['Suppliers']
print(f"Suppliers: {len(sup)} IDs (S001–S150) with a name column only ('Supplier 1' … 'Supplier 150'). No supplier attribute, no region, no category and — critically — no SupplierID on Products, Inventory or Shipments.")
print('Univariate analysis of suppliers is therefore limited to counting them. Supplier performance, reliability and segmentation CANNOT be computed from this dataset;')
print('Part 5 substitutes a Warehouse Reliability Score and Part 7 states this as a data gap with a concrete fix.')
R['n_suppliers'] = len(sup)

# %% [md]
# ## 3.2 Bivariate analysis

# %%
cust = master.groupby('CustomerID').agg(orders=('OrderID', 'size')).reindex(cl['Customers'].CustomerID, fill_value=0)
cv = conf.groupby('CustomerID').order_value.sum().reindex(cl['Customers'].CustomerID, fill_value=0).sort_values()
lor = np.cumsum(cv.values) / cv.sum(); x = np.arange(1, len(cv)+1) / len(cv); gini = 1 - 2*np.trapezoid(lor, x)
fig, axs = plt.subplots(1, 2, figsize=(11, 4)); vc = cust.orders.value_counts().sort_index(); axs[0].bar(vc.index, vc.values, color=BLUE, width=.7)
axs[0].set_title('Customers by number of orders'); axs[0].set_xlabel('Orders placed (Jan–8 Sep 2026)'); axs[0].set_ylabel('Customers')
axs[1].plot(x, lor, color=BLUE, lw=2.5); axs[1].plot([0, 1], [0, 1], color=GREY, ls='--'); axs[1].set_title(f'Customer revenue concentration (Gini {gini:.2f})'); axs[1].set_xlabel('Share of customers (lowest → highest value)'); axs[1].set_ylabel('Cumulative share of confirmed revenue')
top20 = cv.sort_values(ascending=False).head(int(.2*len(cv))).sum() / cv.sum(); R['cust_gini'] = float(gini); R['cust_top20_share'] = float(top20); R['orders_per_cust_mean'] = float(cust.orders.mean()); R['cust_zero_orders'] = int((cust.orders == 0).sum())
chart('10_customer_orders_concentration', 'Customer order frequency and revenue concentration', f"Customers place {cust.orders.mean():.1f} orders on average (max {cust.orders.max()}); {R['cust_zero_orders']} customers never ordered. Revenue is only mildly concentrated: the top 20% of customers deliver {pct(top20)} of confirmed revenue (Gini {gini:.2f}; the Pareto 80/20 rule does not hold).",
      'There is no small group of "whales" to protect — value comes from a broad, evenly-spread base. Retention and frequency programmes across the whole base will matter more than key-account management.')

# %%
tc_ = conf.groupby('CustomerID').agg(rev=('order_value', 'sum'), orders=('OrderID', 'size'), city=('CustomerCity', 'first')).sort_values('rev', ascending=False).head(10)
fig, ax = plt.subplots(figsize=(8.5, 4.2)); ax.barh(tc_.index[::-1], tc_.rev[::-1], color=BLUE, height=.6)
for i, (c, r) in enumerate(tc_[::-1].iterrows()): ax.text(r.rev + 500, i, f'{short(r.rev)} · {int(r.orders)} orders · {r.city}', va='center', fontsize=8.5)
ax.set_title('Top 10 customers by confirmed revenue'); ax.set_xlabel('Confirmed order value (currency units)'); ax.set_ylabel('Customer'); ax.set_xlim(0, tc_.rev.max()*1.5)
R['top_customers'] = [dict(customer=c, revenue=float(r.rev), orders=int(r.orders), city=r.city) for c, r in tc_.iterrows()]; R['top10_cust_share'] = float(tc_.rev.sum() / conf.order_value.sum())
chart('11_top_customers', 'Top 10 customers', f"The top customer ({tc_.index[0]}) generated {short(tc_.rev.iloc[0])}; together the top 10 account for only {pct(R['top10_cust_share'])} of confirmed revenue.",
      'Even SwiftLog\'s best customers are individually small, which limits single-customer risk but also means growth must come from the base, not a few accounts. Use these 10 as a pilot group for loyalty and service-level offers.')

# %%
cat = oi_conf.groupby('Category').agg(revenue=('LineValue', 'sum'), units=('Quantity', 'sum'), products=('ProductID', 'nunique')).assign(share=lambda d: d.revenue/d.revenue.sum()); cat['rev_per_product'] = cat.revenue / cat.products
cat = cat.sort_values('revenue', ascending=False); R['category_revenue'] = cat.round(3).to_dict('index')
fig, axs = plt.subplots(1, 2, figsize=(11, 3.9)); axs[0].bar(cat.index, cat.revenue, color=BLUE, width=.55)
for i, (c, r) in enumerate(cat.iterrows()): axs[0].text(i, r.revenue + 6e4, f'{short(r.revenue)}\n{pct(r.share,0)}', ha='center', fontsize=9)
axs[0].set_title('Confirmed revenue by category'); axs[0].set_ylabel('Revenue (currency units)'); axs[0].set_xlabel('Product category'); axs[0].set_ylim(0, cat.revenue.max()*1.2)
axs[1].bar(cat.index, cat.rev_per_product, color=AQUA, width=.55); axs[1].set_title('Revenue per product listed'); axs[1].set_ylabel('Revenue ÷ products sold'); axs[1].set_xlabel('Product category')
for i, v in enumerate(cat.rev_per_product): axs[1].text(i, v + 300, short(v), ha='center', fontsize=9)
chart('12_category_revenue', 'Category vs revenue', f"{cat.index[0]} leads with {pct(cat.share.iloc[0])} of confirmed revenue ({short(cat.revenue.iloc[0])}), then {cat.index[1]} ({pct(cat.share.iloc[1])}) and {cat.index[2]} ({pct(cat.share.iloc[2])}). On a per-product basis {cat.rev_per_product.idxmax()} is strongest.",
      f"Fashion leads partly because it has the most SKUs (95 vs 77–78); on a per-SKU basis {cat.rev_per_product.idxmax()} is the most productive range. Range-extension decisions should use revenue per SKU, not total revenue.")

# %%
pr = oi_conf.groupby('ProductID').LineValue.sum().sort_values(ascending=False); cum = pr.cumsum() / pr.sum()
n80 = int((cum < .8).sum() + 1); share20 = pr.head(int(.2*len(pr))).sum() / pr.sum()
fig, ax = plt.subplots(figsize=(8.5, 4.2)); ax.plot(np.arange(1, len(cum)+1), cum.values, color=BLUE, lw=2.5); ax.plot([0, len(cum)], [0, 1], color=GREY, ls='--', lw=1)
ax.axhline(.8, color=ORANGE, ls=':'); ax.text(5, .82, f'80% of revenue needs {n80} of {len(pr)} products ({pct(n80/len(pr),0)})', color=ORANGE, fontsize=9)
ax.set_title('Product revenue concentration (Pareto)'); ax.set_xlabel('Products ranked by revenue'); ax.set_ylabel('Cumulative share of confirmed revenue')
R['products_for_80'] = n80; R['top20pct_products_share'] = float(share20); R['top_products'] = [dict(product=p, revenue=float(v), category=cl['Products'].set_index('ProductID').Category[p]) for p, v in pr.head(10).items()]
chart('13_product_pareto', 'Product Pareto curve', f"Revenue is spread thinly across the catalogue: the top 20% of products generate only {pct(share20)} of revenue, and {n80} of {len(pr)} products ({pct(n80/len(pr),0)}) are needed to reach 80%.",
      'The catalogue has no "hero" products and no obvious tail to cut. Best-seller lists (e.g. ' + ', '.join(pr.index[:3]) + ') are only marginally ahead, so range-rationalisation should be based on margin and return data, not revenue rank alone.')

# %%
wi = inv_c.groupby('WarehouseID').agg(stock=('Stock', 'sum'), low=('LowStock', 'sum'), positions=('ProductID', 'size')).join(emp); wi['stock_per_emp'] = wi.stock / wi.employees
R['wh_inventory'] = wi.round(2).to_dict('index')
fig, axs = plt.subplots(1, 2, figsize=(12, 4)); axs[0].bar(wi.index, wi.stock/1000, color=BLUE, width=.65); axs[0].set_title('Total units in stock by warehouse'); axs[0].set_xlabel('Warehouse'); axs[0].set_ylabel('Units (thousands)')
axs[1].bar(wi.index, wi.low, color=[ORANGE if v >= 5 else BLUE for v in wi.low], width=.65)
for i, v in enumerate(wi.low): axs[1].text(i, v + .1, int(v), ha='center', fontsize=9)
axs[1].set_title(f'Low-stock positions (<{LOW_STOCK} units)'); axs[1].set_xlabel('Warehouse'); axs[1].set_ylabel('Positions below threshold')
worst = wi.low.nlargest(3).index.tolist(); R['low_stock_worst'] = worst
chart('14_warehouse_inventory', 'Warehouse vs inventory levels', f"Stock held ranges from {short(wi.stock.min())} ({wi.stock.idxmin()}) to {short(wi.stock.max())} ({wi.stock.idxmax()}) units. Low-stock positions cluster in {', '.join(worst)} ({int(wi.low[worst].sum())} of {int(wi.low.sum())}).",
      'Low-stock positions are concentrated in a handful of warehouses (a few sites hold over a third of them), so replenishment should be targeted there first rather than raised network-wide.')

# %%
dw = master.groupby('WarehouseID').agg(n=('OrderID', 'size'), late=('IsDelayed', 'sum'), md=('DelayDays', 'mean')); dw['rate'] = dw.late / dw.n
ci = np.array([wilson(k, n) for k, n in zip(dw.late, dw.n)]); net = master.IsDelayed.mean()
chi_w = stats.chi2_contingency(pd.crosstab(master.WarehouseID, master.IsDelayed))[1]; R['p_delay_wh'] = float(chi_w); R['wh_delay'] = dw.round(4).to_dict('index')
fig, ax = plt.subplots(figsize=(10.5, 4.3)); ax.bar(dw.index, dw.rate*100, color=BLUE, width=.65, yerr=[(dw.rate-ci[:, 0])*100, (ci[:, 1]-dw.rate)*100], ecolor='#52514e', capsize=3)
ax.axhline(net*100, color=ORANGE, lw=2, ls='--'); ax.text(14.4, net*100 + 1.2, f'network {pct(net)}', ha='right', color=ORANGE, fontsize=9.5); ax.set_ylim(0, 85)
ax.set_title('Share of shipments delayed (≥3 days) by warehouse, with 95% confidence interval'); ax.set_xlabel('Dispatch warehouse'); ax.set_ylabel('% of shipments delayed')
hi_w, lo_w = dw.rate.idxmax(), dw.rate.idxmin()
chart('15_delay_by_warehouse', 'Delivery performance by warehouse', f"Delay rates run from {pct(dw.rate.min())} ({lo_w}) to {pct(dw.rate.max())} ({hi_w}), but every warehouse's confidence interval overlaps the network average of {pct(net)} (χ² p = {chi_w:.2f}).",
      'Warehouse-to-warehouse differences are statistically indistinguishable from chance. Management should NOT single out a "bad" warehouse; the delay problem is network-wide and points to a shared upstream cause (carrier SLAs, dispatch planning or the promise-date rule).')

# %%
dd = master.groupby('DelayDays').agg(n=('OrderID', 'size'), k=('HasReturn', 'sum')); dd['rate'] = dd.k / dd.n; ci2 = np.array([wilson(k, n) for k, n in zip(dd.k, dd.n)])
fig, ax = plt.subplots(figsize=(8.5, 4)); ax.errorbar(dd.index, dd.rate*100, yerr=[(dd.rate-ci2[:, 0])*100, (ci2[:, 1]-dd.rate)*100], color=BLUE, marker='o', ms=8, lw=2.2, capsize=3)
ax.axhline(master.HasReturn.mean()*100, color=ORANGE, ls='--'); ax.set_ylim(0, 16); ax.set_title('Return rate by shipment delay (days)'); ax.set_xlabel('Delay days'); ax.set_ylabel('% of orders with a return')
rho_dr, p_dr = stats.spearmanr(master.DelayDays, master.HasReturn); R['rho_delay_return'] = float(rho_dr); R['p_delay_return_spearman'] = float(p_dr)
chart('16_delay_vs_returns', 'Delay days vs return rate', f"Return rate does not rise with delay: {pct(dd.rate.iloc[0])} at 0 days vs {pct(dd.rate.iloc[-1])} at 7 days; the correlation is ρ = {rho_dr:.3f} (p = {p_dr:.2f}).",
      'In this data, late delivery does not measurably drive returns, so cutting delays should be justified by service-level and customer-experience goals rather than by an expected fall in returns.')

# %%
reg = pd.concat({'Customer city (destination)': master.groupby('CustomerCity').agg(n=('OrderID', 'size'), late=('IsDelayed', 'sum'), md=('DelayDays', 'mean')),
                 'Warehouse city (dispatch)': master.groupby('WarehouseCity').agg(n=('OrderID', 'size'), late=('IsDelayed', 'sum'), md=('DelayDays', 'mean'))}); reg['rate'] = reg.late / reg.n
p_cc = stats.chi2_contingency(pd.crosstab(master.CustomerCity, master.IsDelayed))[1]; p_wc = stats.chi2_contingency(pd.crosstab(master.WarehouseCity, master.IsDelayed))[1]
R['p_delay_custcity'] = float(p_cc); R['p_delay_whcity'] = float(p_wc); R['region_delay'] = {f'{a}|{b}': dict(rate=float(r.rate), mean_delay=float(r.md), n=int(r.n)) for (a, b), r in reg.iterrows()}
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4), sharey=True)
for ax, key, colr in zip(axs, reg.index.get_level_values(0).unique(), [BLUE, AQUA]):
    d = reg.loc[key].sort_values('rate', ascending=False); ax.bar(d.index, d.rate*100, color=colr, width=.55)
    for i, (c, r) in enumerate(d.iterrows()): ax.text(i, r.rate*100 + .6, f'{r.rate*100:.1f}%\n{r.md:.2f} d', ha='center', fontsize=8.5)
    ax.axhline(net*100, color=ORANGE, ls='--'); ax.set_title(key); ax.set_xlabel('City'); ax.set_ylim(0, 80)
axs[0].set_ylabel('% of shipments delayed (≥3 days)')
worst_city = reg.loc['Customer city (destination)'].rate.idxmax()
chart('17_region_vs_shipment_performance', 'Region vs shipment performance', f"By destination, {worst_city} has the highest delay rate ({pct(reg.loc['Customer city (destination)'].rate.max())}) and Lagos the lowest ({pct(reg.loc['Customer city (destination)'].rate.min())}), but the spread is small (χ² p = {p_cc:.2f}); by dispatch city the rates are within one point of each other (p = {p_wc:.2f}).",
      'No region is a significant delay hotspot. Regional delay "league tables" would mislead; the answer to "which region is worst" is "none reliably" — so there is no case for region-specific carrier changes on this evidence.')

# %%
rt2 = cl['Returns'].merge(master[['OrderID', 'IsDelayed', 'DelayDays']], on='OrderID'); ct = pd.crosstab(rt2.Reason, rt2.IsDelayed, normalize='index')*100; p_rr = stats.chi2_contingency(pd.crosstab(rt2.Reason, rt2.IsDelayed))[1]
R['p_reason_delay'] = float(p_rr); R['late_reason_delayed_share'] = float(ct.loc['Late Delivery', 1]/100)
fig, ax = plt.subplots(figsize=(8.5, 3.8)); ax.barh(ct.index, ct[0], color=BLUE, label='Shipment On Time (≤2 days)', height=.55); ax.barh(ct.index, ct[1], left=ct[0], color=ORANGE, label='Shipment Delayed (≥3 days)', height=.55)
for i, (r, row) in enumerate(ct.iterrows()): ax.text(row[0]/2, i, f'{row[0]:.0f}%', ha='center', va='center', color='white', fontsize=9.5); ax.text(row[0]+row[1]/2, i, f'{row[1]:.0f}%', ha='center', va='center', color='white', fontsize=9.5)
ax.set_title('Return reason vs actual delivery outcome'); ax.set_xlabel('% of return records'); ax.set_ylabel('Reason recorded'); ax.legend(loc='lower center', bbox_to_anchor=(.5, -.4), ncol=2); ax.set_xlim(0, 100)
chart('18_reason_vs_delay', 'Return reason vs delivery outcome', f"Only {pct(R['late_reason_delayed_share'],0)} of returns logged as 'Late Delivery' were actually delayed shipments — about the same as {pct(R['delay_rate'],0)} of all shipments, and no different from 'Damaged' or 'Wrong Item' returns (χ² p = {p_rr:.2f}).",
      "The 'Late Delivery' reason code is not corroborated by the delivery data: it is either entered inconsistently or the delay field does not reflect what customers experienced. Return reason capture needs an audit and a controlled pick-list.")

# %%
mth = master.groupby('YearMonth').agg(gross=('order_value', 'sum')).join(conf.groupby('YearMonth').order_value.sum().rename('confirmed')).join(master[master.Status == 'Delivered'].groupby('YearMonth').order_value.sum().rename('realised'))
R['monthly_revenue'] = mth.round(0).to_dict('index')
fig, ax = plt.subplots(figsize=(9.5, 4.2)); xi = np.arange(len(mth))
for col, c, lab in [('gross', GREY, 'Gross (all orders)'), ('confirmed', BLUE, 'Confirmed (excl. cancelled)'), ('realised', ORANGE, 'Realised (delivered)')]:
    ax.plot(xi[:-1], mth[col].values[:-1]/1e6, color=c, lw=2.4, marker='o', ms=6, label=lab); ax.plot(xi[-2:], mth[col].values[-2:]/1e6, color=c, lw=1.5, ls=':')
ax.set_xticks(xi); ax.set_xticklabels(mth.index); ax.set_title('Monthly order value by revenue definition'); ax.set_xlabel('Order month (2026)'); ax.set_ylabel('Order value (millions)')
ax.legend(loc='upper left', bbox_to_anchor=(0, 1.0), ncol=3, fontsize=8.5); ax.set_ylim(0, 3.4); ax.text(len(mth)-1, mth.gross.iloc[-1]/1e6 + .12, 'partial', ha='center', fontsize=8, color='#52514e')
pk = mth.confirmed.iloc[:-1].idxmax(); R['peak_revenue_month'] = pk
chart('19_monthly_revenue', 'Monthly revenue trend', f"Confirmed revenue averages {short(mth.confirmed.iloc[:-1].mean())} a month, peaking in {pk} ({short(mth.confirmed.max())}); there is no sustained growth trend across Jan–Aug. Realised (delivered) value is only {pct(R['rev_realised']/R['rev_gross'],0)} of gross.",
      f"Demand is flat, not growing, and the gap between gross and realised value (≈{short(R['rev_gross']-R['rev_realised'])}) is the real prize. September is partial (to 8 Sep) and must not be read as a decline.")

# %% [md]
# ## 3.3 Multivariate analysis — relationships among several variables affecting logistics performance

# %%
nc = ['units', 'order_value', 'n_lines', 'n_categories', 'DelayDays', 'IsDelayed', 'HasReturn', 'n_returns', 'SameCityDispatch']
cm = master.loc[master.HasItems == 1, nc].corr(method='spearman')
fig, ax = plt.subplots(figsize=(8.2, 6.4)); sns.heatmap(cm, cmap='RdBu_r', center=0, vmin=-1, vmax=1, annot=True, fmt='.2f', annot_kws={'size': 8}, linewidths=.5, ax=ax, cbar_kws={'label': 'Spearman ρ'}); ax.grid(False)
ax.set_title('Correlation among order, delivery and return variables'); ax.set_xlabel('Variable'); ax.set_ylabel('Variable')
off = cm.where(~np.eye(len(cm), dtype=bool)).abs(); structural = {('units', 'order_value'), ('order_value', 'units'), ('units', 'n_lines'), ('n_lines', 'units'), ('order_value', 'n_lines'), ('n_lines', 'order_value'), ('DelayDays', 'IsDelayed'), ('IsDelayed', 'DelayDays'), ('HasReturn', 'n_returns'), ('n_returns', 'HasReturn'), ('n_categories', 'n_lines'), ('n_lines', 'n_categories'), ('n_categories', 'units'), ('units', 'n_categories'), ('n_categories', 'order_value'), ('order_value', 'n_categories')}
cross = off.stack(); cross = cross[[i not in structural for i in cross.index]]; mx = cross.max(); mxi = cross.idxmax(); R['max_nonstructural_corr'] = float(mx)
chart('20_correlation_heatmap', 'Correlation heatmap', f"Strong correlations are purely structural (units ↔ order value ↔ lines, delay days ↔ delay flag). Excluding those, the largest relationship between any operational variable pair is |ρ| = {mx:.2f} ({mxi[0]} vs {mxi[1]}) — effectively zero.",
      'Delivery delay, returns and basket characteristics are statistically unrelated in this dataset. That caps the predictive power any model can extract from these fields and tells SwiftLog that better predictors (carrier, route, dispatch/promise dates, picker, product-level defect flags) must be captured.')

# %%
hm = master.pivot_table(index='WarehouseID', columns='CustomerCity', values='IsDelayed', aggfunc='mean')*100; hn = master.pivot_table(index='WarehouseID', columns='CustomerCity', values='IsDelayed', aggfunc='size')
fig, ax = plt.subplots(figsize=(7.5, 7)); sns.heatmap(hm, annot=True, fmt='.0f', cmap='YlOrBr', vmin=45, vmax=80, linewidths=.5, ax=ax, cbar_kws={'label': '% shipments delayed'}); ax.grid(False)
ax.set_title('Delay rate: dispatch warehouse × destination city'); ax.set_xlabel('Destination (customer city)'); ax.set_ylabel('Dispatch warehouse')
R['route_cell_n_median'] = float(hn.stack().median()); R['route_delay_min'] = float(hm.min().min()); R['route_delay_max'] = float(hm.max().max())
chart('21_route_delay_heatmap', 'Warehouse × destination delay heatmap', f"Across the 60 warehouse–city lanes the delay rate ranges from {hm.min().min():.0f}% to {hm.max().max():.0f}%, but each lane has only ~{hn.stack().median():.0f} shipments, so most of the spread is sampling noise (expected ±{1.96*np.sqrt(.64*.36/hn.stack().median())*100:.0f} pts).",
      'Lane-level league tables are unreliable at this volume. Before acting on any individual lane, accumulate more shipments or pool lanes (e.g. by region). Monitor with control limits rather than raw ranks.')

# %%
ws = wl.join(wi[['low', 'stock']]).join(dw[['rate']]); ws['low'] = ws.low.astype(int)
rho_ld, p_ld = stats.spearmanr(ws.ship_per_emp, ws.rate); rho_ll, p_ll = stats.spearmanr(ws.ship_per_emp, ws.low); R['rho_load_delay'] = float(rho_ld); R['p_load_delay'] = float(p_ld); R['rho_load_low'] = float(rho_ll); R['p_load_low'] = float(p_ll)
fig, ax = plt.subplots(figsize=(8.5, 5.2)); sc = ax.scatter(ws.ship_per_emp, ws.rate*100, s=60 + ws.low*70, c=ws.stock/1000, cmap='Blues', edgecolor=BLUE, lw=1.2, vmin=40, vmax=100)
for w, r in ws.iterrows(): ax.annotate(w, (r.ship_per_emp, r.rate*100), xytext=(6, 5), textcoords='offset points', fontsize=8.5)
ax.set_title('Warehouse workload vs delay rate (bubble = low-stock positions, shade = stock)'); ax.set_xlabel('Shipments per employee'); ax.set_ylabel('% shipments delayed'); plt.colorbar(sc, label='Units in stock (thousands)')
chart('22_workload_delay_stock_bubble', 'Workload, delay and stock by warehouse', f"Heavier workload per employee is not associated with more delays (ρ = {rho_ld:.2f}, p = {p_ld:.2f}, n = 15) and only weakly with low-stock positions (ρ = {rho_ll:.2f}, p = {p_ll:.2f}; not significant).",
      'The link between workload and low stock is weak and not statistically significant with only 15 warehouses, so it should be monitored rather than acted on. Workload does not explain delays at all, which again points to network-wide causes.')

# %%
sc_ = master.groupby('Status').agg(n=('OrderID', 'size'), delay=('IsDelayed', 'mean'), ret=('HasReturn', 'mean'), value=('order_value', 'sum'))
R['status_table'] = sc_.round(4).to_dict('index'); R['returns_on_nondelivered'] = int(R['dq']['Returns recorded against non-Delivered orders']); R['ship_on_cancelled'] = int(R['dq']['Shipments recorded for CANCELLED orders'])
fig, ax = plt.subplots(figsize=(8.5, 3.9)); xx = np.arange(3); w_ = .36; ax.bar(xx - w_/2, sc_.delay*100, w_, color=ORANGE, label='% shipments delayed'); ax.bar(xx + w_/2, sc_.ret*100, w_, color=BLUE, label='% orders returned')
for i, (s, r) in enumerate(sc_.iterrows()): ax.text(i - w_/2, r.delay*100 + 1, f'{r.delay*100:.0f}%', ha='center', fontsize=9); ax.text(i + w_/2, r.ret*100 + 1, f'{r.ret*100:.1f}%', ha='center', fontsize=9)
ax.set_xticks(xx); ax.set_xticklabels(sc_.index); ax.set_title('Delay and return rates by order status — a process-integrity test'); ax.set_xlabel('Order status'); ax.set_ylabel('Rate (%)'); ax.legend(); ax.set_ylim(0, 80)
chart('23_status_integrity', 'Order status vs delay and returns', f"Cancelled and Pending orders show the same delay ({pct(sc_.delay['Cancelled'],0)} / {pct(sc_.delay['Pending'],0)}) and return rates ({pct(sc_.ret['Cancelled'])} / {pct(sc_.ret['Pending'])}) as Delivered orders: {R['ship_on_cancelled']:,} shipments and {int(cl['Returns'].merge(master[['OrderID','Status']], on='OrderID').query('Status==\"Cancelled\"').shape[0])} returns are recorded against Cancelled orders.",
      'A cancelled order cannot be shipped late or returned. Either status is not being updated (systems out of sync) or shipment/return records are mis-keyed. Until fixed, every delivery KPI is contaminated — reconciling order status with shipment and return events is a prerequisite for trustworthy analytics and ML.')
print(f'\nTotal charts produced so far: {len(CH)} (brief requires at least 15).')
