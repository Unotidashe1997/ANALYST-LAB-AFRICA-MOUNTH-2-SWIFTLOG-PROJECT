# %% [md]
# ---
# # PART 4 — Statistical Analysis
# Tests use α = 0.05 and are reported with an **effect size**. Because we run several tests, a Bonferroni-adjusted threshold (α/k) is also shown: a result only counts as robust if it survives it.

# %%
num = master.loc[master.HasItems == 1, ['units', 'order_value', 'n_lines', 'DelayDays']]
ds = num.describe().T; ds['skew'] = num.skew(); ds['kurtosis'] = num.kurt(); ds['cv'] = ds['std'] / ds['mean']
R['desc_stats'] = ds.round(3).to_dict('index')
ds.round(2)

# %%
nt = pd.DataFrame({v: dict(shapiro_p=stats.shapiro(num[v].sample(min(len(num), 4000), random_state=1))[1], skew=stats.skew(num[v])) for v in num}).T
print('Normality (Shapiro–Wilk on up to 4,000 rows): p < 0.05 means NOT normal. Right-skewed order value and units favour non-parametric tests (Mann–Whitney, Kruskal–Wallis, Spearman).')
nt.round(4)

# %%
def cramers_v(ct):
    chi2 = stats.chi2_contingency(ct, correction=False)[0]; n = ct.values.sum(); return float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))
tests = []
def add(name, h0, test, stat, p, eff_name, eff):
    tests.append(dict(test=name, H0=h0, method=test, statistic=round(float(stat), 3), p_value=float(p), effect_size=f'{eff_name} = {eff:.3f}'))
ct = pd.crosstab(master.WarehouseID, master.IsDelayed); c = stats.chi2_contingency(ct); add('T1 Delay vs warehouse', 'Delay rate is equal across the 15 warehouses', 'Chi-square independence', c[0], c[1], "Cramér's V", cramers_v(ct))
ct = pd.crosstab(master.CustomerCity, master.IsDelayed); c = stats.chi2_contingency(ct); add('T2 Delay vs destination city', 'Delay rate is equal across the 4 cities', 'Chi-square independence', c[0], c[1], "Cramér's V", cramers_v(ct))
ct = pd.crosstab(master.IsDelayed, master.HasReturn); c = stats.chi2_contingency(ct); add('T3 Returns vs delay', 'Return rate is the same for delayed and on-time orders', 'Chi-square independence', c[0], c[1], "Cramér's V", cramers_v(ct))
a, b = master.loc[(master.IsDelayed == 1) & (master.HasItems == 1), 'order_value'], master.loc[(master.IsDelayed == 0) & (master.HasItems == 1), 'order_value']
u = stats.mannwhitneyu(a, b); add('T4 Order value: delayed vs on-time', 'Order value distributions are equal', 'Mann–Whitney U', u[0], u[1], 'rank-biserial r', 1 - 2*u[0]/(len(a)*len(b)))
g = [d.order_value.values for _, d in master[master.HasItems == 1].groupby('DominantCategory')]; k = stats.kruskal(*g); n_ = sum(map(len, g)); add('T5 Order value by dominant category', 'Order value is equal across categories', 'Kruskal–Wallis', k[0], k[1], 'epsilon²', k[0] / (n_ - 1))
g = [d.order_value.values for _, d in master[master.HasItems == 1].groupby('Status')]; k = stats.kruskal(*g); add('T6 Order value by status', 'Order value is equal across statuses', 'Kruskal–Wallis', k[0], k[1], 'epsilon²', k[0] / (n_ - 1))
ct = pd.crosstab(master.SameCityDispatch, master.IsDelayed); c = stats.chi2_contingency(ct); add('T7 Delay: same-city vs cross-city dispatch', 'Delay rate is equal whether or not warehouse and customer are in the same city', 'Chi-square independence', c[0], c[1], "Cramér's V", cramers_v(ct))
ct = pd.crosstab(master.CustomerCity, master.Status); c = stats.chi2_contingency(ct); add('T8 Cancellation vs destination city', 'Order status mix is equal across cities', 'Chi-square independence', c[0], c[1], "Cramér's V", cramers_v(ct))
r_, p_ = stats.spearmanr(master.units[master.HasItems == 1], master.DelayDays[master.HasItems == 1]); add('T9 Basket size vs delay days', 'No monotonic relationship', 'Spearman correlation', r_, p_, 'rho', r_)
ct = pd.crosstab(master.WarehouseID, master.HasReturn); c = stats.chi2_contingency(ct); add('T10 Returns vs warehouse', 'Return rate is equal across warehouses', 'Chi-square independence', c[0], c[1], "Cramér's V", cramers_v(ct))
T = pd.DataFrame(tests); alpha_b = 0.05 / len(T)
T['significant_0.05'] = np.where(T.p_value < .05, 'Yes', 'No'); T['survives_Bonferroni'] = np.where(T.p_value < alpha_b, 'Yes', 'No'); T['p_value'] = T.p_value.round(4)
R['tests'] = T.to_dict('records'); R['bonferroni_alpha'] = alpha_b; R['n_sig_raw'] = int((T.p_value < .05).sum()); R['n_sig_bonf'] = int((T.p_value < alpha_b).sum())
print(f'{len(T)} tests run. Significant at 0.05: {R["n_sig_raw"]}; surviving Bonferroni (α = {alpha_b:.4f}): {R["n_sig_bonf"]}.')
T[['test', 'method', 'statistic', 'p_value', 'effect_size', 'significant_0.05', 'survives_Bonferroni']]

# %%
for _, t in T.iterrows():
    verdict = ('Statistically significant' + (' and survives the Bonferroni correction.' if t.survives_Bonferroni == 'Yes' else ' at 0.05 but NOT after Bonferroni correction — treat as a possible false positive.')) if t['significant_0.05'] == 'Yes' else 'No significant difference — we fail to reject H0; the observed gap is consistent with chance.'
    print(f"{t.test}\n  H0: {t.H0}\n  Result: {t.method}, p = {t.p_value:.4f}, {t.effect_size}. {verdict}\n")
print('Business reading: delivery delay, returns, order value and warehouse/region show no reliable relationship in this extract. The operational story is therefore a network-wide delay level (63.7%), not a "problem warehouse" or "problem region".')

# %%
lo, hi = wilson(int(master.IsDelayed.sum()), len(master)); lo2, hi2 = wilson(int(master.HasReturn.sum()), len(master))
ci_tab = pd.DataFrame({'metric': ['Delayed shipments (≥3 days)', 'Orders with a return'], 'estimate': [master.IsDelayed.mean(), master.HasReturn.mean()], 'ci_low': [lo, lo2], 'ci_high': [hi, hi2]})
R['delay_ci'] = [float(lo), float(hi)]; R['return_ci'] = [float(lo2), float(hi2)]
ci_tab.round(4)
