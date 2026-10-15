# %% [md]
# ---
# # PART 7 — Business Insights & Recommendations
# Each answer cites the evidence (chart/test) it rests on. Where the data cannot answer a question, we say so and state what must be captured.

# %%
# Q5 evidence: product-level return rate (share of order-lines whose order had a return), with a noise check
ol = cl['Order_Items'].merge(master[['OrderID', 'HasReturn', 'Status']], on='OrderID'); pr_ = ol.groupby('ProductID').agg(lines=('OrderID', 'size'), rr=('HasReturn', 'mean')).join(cl['Products'].set_index('ProductID').Category)
pr_ = pr_[pr_.lines >= 30]; base_rr = ol.HasReturn.mean(); se = np.sqrt(base_rr * (1 - base_rr) / pr_.lines); pr_['z'] = (pr_.rr - base_rr) / se
n_out = int((pr_.z.abs() > 3).sum()); exp_out = len(pr_) * 0.0027
top_ret = pr_.sort_values('rr', ascending=False).head(5)
cat_rr = ol.merge(cl['Products'][['ProductID', 'Category']], on='ProductID', suffixes=('', '_p')).groupby('Category_p').HasReturn.mean() if 'Category' in ol else None
ctp = pd.crosstab(ol.merge(cl['Products'][['ProductID', 'Category']].rename(columns={'Category': 'Cat'}), on='ProductID').Cat, ol.HasReturn); p_cat_ret = stats.chi2_contingency(ctp)[1]
R['prod_ret_top'] = top_ret.round(3).to_dict('index'); R['prod_ret_outliers'] = n_out; R['prod_ret_expected_outliers'] = float(exp_out); R['p_cat_return'] = float(p_cat_ret); R['base_product_return_rate'] = float(base_rr)
print(f'Products with ≥30 order lines: {len(pr_)}. Highest return rate: {top_ret.index[0]} at {pct(top_ret.rr.iloc[0])} (n={int(top_ret.lines.iloc[0])} lines) vs network {pct(base_rr)}.')
print(f'Products more than 3 SE away from the network rate: {n_out} (≈{exp_out:.1f} expected by chance). Return rate by category: p = {p_cat_ret:.2f}.')
top_ret.round(3)

# %%
cr = R['category_revenue']; topcat = max(cr, key=lambda k: cr[k]['revenue']); seg = R['segments']
champ = max(seg.values(), key=lambda v: v['share_of_confirmed_revenue']); atrisk = [v for v in seg.values() if v['segment'].startswith('At-risk')]
wr = R['wh_reliability']; best_w = max(wr, key=lambda k: wr[k]['reliability_score']); worst_w = min(wr, key=lambda k: wr[k]['reliability_score'])
ws_ = sorted(R['wh_load'], key=lambda k: -R['wh_load'][k]['ship_per_emp'])[:3]; lowst = R['low_stock_worst']
rd = {k: v for k, v in R['region_delay'].items() if k.startswith('Customer city')}; worst_reg = max(rd, key=lambda k: rd[k]['rate']); best_reg = min(rd, key=lambda k: rd[k]['rate'])
A = {}
A['1. Which customer segments generate the highest business value?'] = (f"{champ['segment']} — {champ['customers']:.0f} customers averaging {short(champ['monetary'])} each, {pct(champ['share_of_confirmed_revenue'],0)} of confirmed revenue. A second value pool is the lapsing high-value group ({sum(v['customers'] for v in atrisk):.0f} customers, {pct(sum(v['share_of_confirmed_revenue'] for v in atrisk),0)} of revenue, last order ≈{atrisk[0]['recency']:.0f} days ago). Revenue is not concentrated in a few accounts (top 20% of customers = {pct(R['cust_top20_share'],0)}, Gini {R['cust_gini']:.2f}). Evidence: charts 10, 11, 26.")
A['2. Which products contribute most to revenue?'] = (f"{topcat} is the top category ({pct(cr[topcat]['share'],0)} of confirmed revenue, partly because it has the most SKUs); the top products are {', '.join(p['product'] for p in R['top_products'][:3])}. But concentration is low: top 20% of products = {pct(R['top20pct_products_share'],0)} of revenue and {R['products_for_80']} of 250 products are needed for 80%. Rankings also shift with the status filter (only {R['top10_overlap_conf_vs_delivered']}/10 top products are the same on delivered-only revenue). Evidence: charts 12, 13, 27.")
A['3. Which suppliers consistently underperform?'] = ("This cannot be answered. The Suppliers table lists 150 IDs and names only; no SupplierID exists on Products, Inventory, Shipments or Orders, so no delivery, quality or return outcome can be traced to a supplier. We did not fabricate a link. Substitute: warehouse reliability score (chart 28). Fix: add SupplierID to Products/Inventory receipts and capture PO promised vs received dates.")
A['4. Which warehouses require operational improvements?'] = (f"Delay differences between warehouses are not statistically significant (χ² p = {R['p_delay_wh']:.2f}), so no warehouse can be blamed for delays. The operational gaps are staffing and stock: highest shipments per employee are {', '.join(ws_)} (up to {R['wh_load'][ws_[0]]['ship_per_emp']:.0f} vs {min(v['ship_per_emp'] for v in R['wh_load'].values()):.0f}), and low-stock positions cluster in {', '.join(lowst)}. Composite reliability: best {best_w}, weakest {worst_w}. Evidence: charts 8, 14, 15, 22, 28.")
A['5. Which products have the highest return rates?'] = (f"Highest observed: {', '.join(top_ret.index[:3])} (up to {pct(top_ret.rr.iloc[0])} vs {pct(base_rr)} overall), but with ~{pr_.lines.median():.0f} lines per product this is mostly noise: {n_out} products exceed ±3 SE (≈{exp_out:.1f} expected by chance) and category does not matter (p = {p_cat_ret:.2f}). Reason mix: {', '.join(f'{k} {v}' for k, v in R['return_reasons'].items())}. No product should be delisted on return rate alone. Evidence: chart 7.")
A['6. Which regions experience the greatest delivery delays?'] = (f"{worst_reg.split('|')[1]} has the highest delay rate ({pct(rd[worst_reg]['rate'])}) and {best_reg.split('|')[1]} the lowest ({pct(rd[best_reg]['rate'])}), but the difference is not significant (p = {R['p_delay_custcity']:.2f}). Honest answer: no region is reliably worse; the network average is {pct(R['delay_rate'])}. Evidence: charts 17, 21.")
A['7. What operational bottlenecks have you identified?'] = (f"(1) Network-wide lateness: {pct(R['delay_rate'])} of shipments are ≥3 days late and delay is uniform 0–7 days. (2) Order conversion: only {pct(R['delivered_rate'])} of orders are Delivered; {pct(R['cancel_rate'])} Cancelled ({short(R['cancelled_value'])}) and {pct(R['pending_rate'])} still Pending. (3) Process integrity: {R['ship_on_cancelled']:,} shipments are recorded against Cancelled orders. (4) Staffing imbalance ({R['load_ratio']:.1f}× workload spread). (5) Order-capture gaps: {R['orders_no_items']} orders without line items. Evidence: charts 3, 6, 8, 23.")
A['8. Which variables are most suitable for predictive modelling?'] = (f"Candidate targets: IsDelayed, DelayDays, HasReturn. Engineered features (customer history, warehouse rolling delay, load, basket) were tested, but the best time-split AUC was {R['best_auc_delay']:.2f} for delay and {R['best_auc_return']:.2f} for returns — essentially chance; {R['feature_screen_nsig']} of 24 features pass p<0.05 (≈2 by chance). So current variables are NOT suitable for delay/return prediction. Revenue, order volume and customer recency/frequency are the modelling-ready quantities (forecasting, churn). Evidence: chart 24.")
A['9. What hidden patterns did your analysis uncover?'] = (f"(a) Statuses do not drive outcomes: Cancelled and Pending orders have the same delay/return rates as Delivered ones, implying the status field and shipment/return events are out of sync. (b) 'Late Delivery' returns are no more likely to be actually delayed ({pct(R['late_reason_delayed_share'],0)}) than any other return. (c) Delay is unrelated to value, basket, route, weekday or returns (max |ρ| {R['max_nonstructural_corr']:.2f}). (d) Revenue is flat across Jan–Aug with no growth trend. (e) Ibadan has the largest customer base but no warehouse. Evidence: charts 18, 20, 23.")
for k, v in A.items(): print(k, '\n   ', v, '\n')
R['answers'] = A

# %%
recs = [
 ('Reconcile order status with shipment and return events', 'Data / Process', f"{R['ship_on_cancelled']:,} shipments and 161 returns sit on Cancelled orders; delay KPIs are contaminated", 'High', 'Trusted KPIs; prerequisite for any ML'),
 ('Launch a network-wide on-time delivery programme (carrier SLAs, dispatch cut-offs, promise-date review)', 'Operations', f"{pct(R['delay_rate'])} late, uniform across sites/regions — a systemic cause", 'High', 'Each 10-point cut in lateness lifts on-time rate to ~46%'),
 ('Attack the cancellation/pending leak with a funnel review (stock confirmation, payment, address checks)', 'Commercial', f"{pct(R['cancel_rate'])} cancelled, {short(R['cancelled_value'])} order value; only {pct(R['delivered_rate'])} delivered", 'High', 'Recovering 10% of cancelled value ≈ ' + short(R['cancelled_value'] * .1)),
 ('Rebalance warehouse staffing toward overloaded sites (' + ', '.join(ws_) + ')', 'Operations', f"{R['load_ratio']:.1f}× spread in shipments per employee with near-equal volumes", 'Medium', 'Better utilisation without extra headcount'),
 ('Targeted replenishment at low-stock hotspots (' + ', '.join(lowst) + ')', 'Inventory', f"{R['low_stock_positions']} positions below {LOW_STOCK} units, concentrated in few sites", 'Medium', 'Fewer stock-outs without raising total stock'),
 ('Win-back programme for lapsing high-value customers', 'Customer', f"{sum(v['customers'] for v in atrisk):.0f} customers, {pct(sum(v['share_of_confirmed_revenue'] for v in atrisk),0)} of revenue, ≈{atrisk[0]['recency']:.0f} days since last order", 'High', 'Protects about a third of confirmed revenue'),
 ('Introduce a loyalty/frequency scheme across the whole base rather than key-account management', 'Customer', f"Top 20% of customers = {pct(R['cust_top20_share'],0)} of revenue (Gini {R['cust_gini']:.2f}); no whales", 'Medium', 'Raises orders per customer (now ' + f"{R['orders_per_cust_mean']:.1f})"),
 ('Assess an Ibadan fulfilment point', 'Network', 'Largest customer city (26%) with no warehouse; all Ibadan orders ship cross-city', 'Medium', 'Possible shorter lead times; needs cost-benefit (distance data not available)'),
 ('Capture the missing data: SupplierID, carrier, promised/dispatch/delivery dates, warehouse capacity, return product', 'Data', 'Supplier performance, capacity utilisation and ML are impossible today', 'High', 'Unlocks supplier scorecards and delay prediction'),
 ('Fix order capture: block orders without line items and fix the empty-basket cohort', 'Data / Process', f"{R['orders_no_items']} orders ({pct(R['orders_no_items']/5000)}) have no items; {R['rfm_no_value_customers']} customers with no valued orders", 'Medium', 'Complete revenue visibility'),
 ('Replace free-text return reasons with a controlled pick-list and audit "Late Delivery" coding', 'Quality', f"Only {pct(R['late_reason_delayed_share'],0)} of 'Late Delivery' returns were delayed shipments", 'Medium', 'Reliable root-cause for returns'),
 ('Tackle controllable returns (wrong item, damaged) through pick-accuracy checks and packaging', 'Quality', f"Wrong Item + Damaged = {pct((R['return_reasons']['Wrong Item']+R['return_reasons']['Damaged'])/sum(R['return_reasons'].values()),0)} of returns; {pct(R['return_rate'])} of orders affected", 'Medium', 'Lower return handling cost'),
 ('Manage the range by revenue per SKU and margin, not total revenue; do not cut the tail yet', 'Merchandising', f"{R['products_for_80']} of 250 products needed for 80% of revenue; class C still earns {pct(R['abc']['C (bottom 50%)']['revenue_share'],0)}", 'Low', 'Avoids losing tail revenue'),
 ('Monitor delays with control charts at region level, not warehouse/lane league tables', 'Analytics', f"Lane samples ≈{R['route_cell_n_median']:.0f} shipments; most spread is noise", 'Low', 'Avoids acting on noise'),
 ('Build future ML (delay-risk scoring, demand forecast, churn) only after new data fields exist', 'Analytics', f"Current best AUC {R['best_auc_delay']:.2f}", 'Medium', 'Avoids deploying a useless model'),
]
RC = pd.DataFrame(recs, columns=['recommendation', 'area', 'evidence', 'priority', 'expected_impact']); RC.index = RC.index + 1; R['recommendations'] = RC.reset_index().to_dict('records')
A['10. What recommendations would you make to improve SwiftLog\'s operations?'] = f"{len(RC)} prioritised recommendations — see table below; top priorities: status reconciliation, network-wide on-time programme, cancellation funnel, data capture, win-back."
R['answers'] = A
print(f'{len(RC)} strategic recommendations (brief requires ≥10).')
RC

# %%
lim = ['Order dates cover 1 Jan – 8 Sep 2026; September is partial.', 'No currency, cost, margin, supplier link, carrier, capacity or ship/delivery dates in the data; revenue is order value from unit price × quantity.',
       'Order status is inconsistent with shipments/returns, so "realised" revenue is a conservative lower bound.', 'The synthetic-looking uniformity of delay, returns and values limits predictive and segmentation power; tests are many, so only Bonferroni-surviving findings should drive investment (none did).',
       'Warehouse capacity is unknown, so workload per employee is a proxy not utilisation.']
R['limitations'] = lim
for l in lim: print('•', l)
print(f'\nFinal chart count: {len(CH)}')
