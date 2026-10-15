"""Run AFTER run_pipeline.py (needs Notebook/results.json). From the project root: python Notebook/src/08_build_reports.py"""
import json, textwrap
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle, KeepTogether
from PIL import Image as PI
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; P = str(ROOT) + '/'; D = json.load(open(str(ROOT / 'Notebook' / 'results.json'))); R = D['R']; CH = D['CH']; C = {c['file'][:2]: c for c in CH}
BLUE = colors.HexColor('#2a78d6'); INK = colors.HexColor('#0b0b0b'); MUT = colors.HexColor('#52514e'); LIGHT = colors.HexColor('#f4f6fa')
ss = getSampleStyleSheet()
S = dict(h1=ParagraphStyle('h1', parent=ss['Heading1'], fontName='Helvetica-Bold', fontSize=16, textColor=INK, spaceBefore=6, spaceAfter=8), h2=ParagraphStyle('h2', parent=ss['Heading2'], fontName='Helvetica-Bold', fontSize=12, textColor=BLUE, spaceBefore=8, spaceAfter=4),
         b=ParagraphStyle('b', parent=ss['BodyText'], fontName='Helvetica', fontSize=9.5, leading=13.5, textColor=INK, spaceAfter=5), sm=ParagraphStyle('sm', parent=ss['BodyText'], fontName='Helvetica', fontSize=8, leading=10.5, textColor=MUT),
         cap=ParagraphStyle('cap', parent=ss['BodyText'], fontName='Helvetica-Oblique', fontSize=8.5, leading=11, textColor=MUT, spaceAfter=3), cell=ParagraphStyle('cell', parent=ss['BodyText'], fontName='Helvetica', fontSize=8, leading=10), cellb=ParagraphStyle('cellb', parent=ss['BodyText'], fontName='Helvetica-Bold', fontSize=8, leading=10, textColor=colors.white),
         ti=ParagraphStyle('ti', parent=ss['Title'], fontName='Helvetica-Bold', fontSize=24, leading=28, textColor=INK, alignment=0), sub=ParagraphStyle('sub', parent=ss['BodyText'], fontName='Helvetica', fontSize=12, leading=16, textColor=MUT))
def esc(t): return str(t).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
def para(t, s='b'): return Paragraph(esc(t), S[s])
def bullets(items): return [Paragraph('&bull; ' + esc(i), ParagraphStyle('bl', parent=S['b'], leftIndent=12, firstLineIndent=-8, spaceAfter=3)) for i in items]
def table(rows, widths, head=True):
    data = [[Paragraph(esc(c), S['cellb'] if (head and i == 0) else S['cell']) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if head else 0); st = [('VALIGN', (0, 0), (-1, -1), 'TOP'), ('GRID', (0, 0), (-1, -1), .4, colors.HexColor('#d9dee8')), ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]
    if head: st.append(('BACKGROUND', (0, 0), (-1, 0), BLUE))
    t.setStyle(TableStyle(st)); return t
def fig(prefix, width=15.5):
    c = C[prefix]; path = P + 'Visualizations/Charts/' + c['file']; w, h = PI.open(path).size; W = width * cm; H = W * h / w
    if H > 9.5 * cm: H = 9.5 * cm; W = H * w / h
    return KeepTogether([Image(path, W, H), Paragraph(f"Figure {int(prefix)}. {esc(c['title'])}. {esc(c['interp'])}", S['cap']), Paragraph('<b>Business implication:</b> ' + esc(c['impl']), S['cap']), Spacer(1, 6)])
def footer(canv, doc): canv.saveState(); canv.setFont('Helvetica', 8); canv.setFillColor(MUT); canv.drawString(2*cm, 1.2*cm, 'AnalystLab Africa Consulting · SwiftLog Mission DS05'); canv.drawRightString(A4[0] - 2*cm, 1.2*cm, f'Page {doc.page}'); canv.restoreState()
f = lambda v: f'{v/1e6:.2f}M'
# =================== TECHNICAL REPORT ===================
st = []
st += [Spacer(1, 4*cm), para('SwiftLog Logistics Limited', 'sub'), Paragraph('Supply Chain Intelligence &amp; Feature Engineering for Predictive Logistics', S['ti']), Spacer(1, 10), para('Technical Report · Mission DS05 · Data Science Track · Week 5', 'sub'), Spacer(1, 16),
       para('Prepared by: Junior Data Scientist (name to be inserted), AnalystLab Africa Consulting', 'sub'), para('Data: 12 SwiftLog tables, 5,000 orders, 1 Jan – 8 Sep 2026', 'sub'), PageBreak()]
st += [Paragraph('1. Executive Summary', S['h1']),
 para(f"SwiftLog asked what its operational data says about supply chain performance and what would be needed for predictive logistics. We cleaned and linked 12 tables (about {R['total_rows']:,} rows), built one order-level analytical table, ran 29 visual analyses, 10 hypothesis tests, 80+ engineered fields, three segmentations and a baseline machine-learning test."),
 para('Headline findings'),] + bullets([
 f"Only {R['delivered_rate']*100:.1f}% of orders are Delivered; {R['cancel_rate']*100:.1f}% are Cancelled ({f(R['cancelled_value'])} of order value) and {R['pending_rate']*100:.1f}% are Pending. Confirmed (non-cancelled) order value is {f(R['rev_confirmed'])} against {f(R['rev_gross'])} gross.",
 f"{R['delay_rate']*100:.1f}% of shipments are delayed by 3 or more days (mean {R['mean_delay']:.2f} days). Delay is spread evenly from 0 to 7 days and is statistically the same across warehouses (p = {R['p_delay_wh']:.2f}), regions (p = {R['p_delay_custcity']:.2f}), weekdays and order types. This is a network-wide problem, not a problem-warehouse issue.",
 f"Order status contradicts the shipment and return records: {R['ship_on_cancelled']:,} shipments sit on Cancelled orders. Delivery KPIs cannot be fully trusted until this is reconciled.",
 f"Revenue is flat (about {f(R['rev_confirmed']/8.3)} a month) and evenly spread: the top 20% of customers deliver {R['cust_top20_share']*100:.0f}% of revenue and no product group dominates.",
 f"Supplier performance cannot be analysed: no table links suppliers to products, stock or shipments. We report this as a data gap rather than inventing a link.",
 f"None of the 10 hypothesis tests was significant at 0.05, and the best predictive model reached AUC {R['best_auc_delay']:.2f} (0.50 is chance). Today's fields cannot predict delays or returns; new data must be captured first."]) + [
 para('Top recommendations: reconcile order status with shipment events; run a network-wide on-time delivery programme; attack the cancellation and pending leak; rebalance warehouse staffing; run a win-back programme for lapsing high-value customers; and capture supplier, carrier and date fields before building ML.'), PageBreak()]
# 2 Data understanding
st += [Paragraph('2. Data Understanding', S['h1']), para(f"The data set contains 12 CSV tables (Customers, Products, Suppliers, Warehouses, Employees, Orders, Order_Items, Inventory, Shipments, Delivery_Performance, Returns, plus supplied Orders). Orders is the hub: customers, line items, shipments and returns hang off OrderID, and warehouses link shipments, stock and staff. The Suppliers table is isolated."), fig('01', 14),
 para(f"Key integrity checks: no missing values in any table; no orphan foreign keys ({R['fk_orphans_total']} found); Delivery_Performance agrees with Shipments on all delay values; and the delivery status is a pure rule of DelayDays (0–2 On Time, 3+ Delayed). The checks below show what the data does not contain.")]
dq = R['dq']; rows = [['Check', 'Result']] + [[k, str(v)] for k, v in dq.items()]
st += [table(rows, [12.5*cm, 4*cm]), Spacer(1, 6), para('A full automated profile of every table is in Reports/Data_Profiling_Report.html. ydata-profiling could not be installed in the build environment (no package index access), so an equivalent profile was generated with custom code: per-column type, missingness, distinct values, summary statistics and distributions.', 'sm'), PageBreak()]
# 3 Cleaning
st += [Paragraph('3. Data Cleaning Process', S['h1']), para('All cleaning lives in reusable functions (standardise_text, parse_dates, iqr_flag, clean_all, build_order_master) so that the same pipeline can be re-run on a new extract. Every step is logged with rows affected and a reason. No rows were deleted other than exact duplicates.')]
rows = [['Step', 'Action', 'Rows', 'Reason'], ['1', 'Trim whitespace, unify case of text/ID fields (all tables)', '0 changed', 'Prevents duplicate categories; data was already clean'], ['2', 'Convert OrderDate to datetime; derive Year, Month, Quarter, DayOfWeek, YearMonth', '5,000', 'Enables time analysis. Last order is 8 Sep 2026, so September is partial'],
 ['3', 'Merge repeated (Order, Product) lines', '96', 'Same product twice in an order is one basket line'], ['4', 'Create LineValue = Quantity × UnitPrice', '14,904', 'Revenue is not stored in the raw data'], ['5', 'Drop exact duplicate Inventory rows', '3', 'Identical repeated records'],
 ['6', 'Sum stock across repeated (Warehouse, Product) keys', '1,542', 'Assumed to be stock lots; sensitivity check gave rank correlation 0.82 so conclusions are largely stable'], ['7', 'IsDelayed = DelayDays ≥ 3; DelaySeverity', '5,000', 'Rule verified against supplied delivery status'],
 ['8', 'Build order-level master table (5,000 × 31)', '5,000', 'Single analytical grain'], ['9', 'Flag IQR outliers; cap order value at the 99th percentile in a separate column', '344 flags', 'Extreme but valid values kept; capped copy for sensitive statistics']]
st += [table(rows, [1.2*cm, 6.8*cm, 1.8*cm, 6.7*cm]), Spacer(1, 6), para(f"Outliers were flagged, not removed: Order_Items.Quantity {R['outliers']['Order_Items.Quantity (consolidated)']}, LineValue {R['outliers']['Order_Items.LineValue']}, order value {R['outliers']['Orders.order_value (orders with items)']}, consolidated stock {R['outliers']['Inventory.Stock (consolidated)']}. They are genuine large orders and stock positions. Categorical encoding is covered in Section 6. Twelve cleaned CSVs are saved in Data/Cleaned_Datasets."), PageBreak()]
# 4 EDA
st += [Paragraph('4. Exploratory Data Analysis', S['h1']), para(f"Revenue definitions: gross = all orders ({f(R['rev_gross'])}); confirmed = non-cancelled ({f(R['rev_confirmed'])}), used as the default; realised = Delivered only ({f(R['rev_realised'])}). The 23 charts below each carry an interpretation and business implication; all 29 charts are in Visualizations/Charts.")]
for k in ['02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']: st.append(fig(k))
st.append(PageBreak())
# 5 Stats
st += [Paragraph('5. Statistical Analysis', S['h1']), para(f"Order value and units are right-skewed, so non-parametric tests were used. Ten hypothesis tests were run at alpha 0.05 with effect sizes. Bonferroni threshold for ten tests: {R['bonferroni_alpha']:.4f}.")]
rows = [['Test', 'Method', 'p-value', 'Effect size', 'Significant?']] + [[t['test'], t['method'], f"{t['p_value']:.3f}", t['effect_size'], t['significant_0.05']] for t in R['tests']]
st += [table(rows, [5.2*cm, 3.6*cm, 1.8*cm, 3.6*cm, 2.3*cm]), Spacer(1, 6), para(f"Result: {R['n_sig_raw']} of 10 tests were significant at 0.05 and {R['n_sig_bonf']} survived correction. Every effect size is negligible (Cramér's V below 0.06). Business reading: delay, returns, order value and warehouse or region show no reliable relationship in this extract. 95% confidence intervals: delayed share {R['delay_ci'][0]*100:.1f}–{R['delay_ci'][1]*100:.1f}%, return share {R['return_ci'][0]*100:.1f}–{R['return_ci'][1]*100:.1f}%."), PageBreak()]
# 6 FE
st += [Paragraph('6. Feature Engineering', S['h1']), para(f"The feature table has {R['fe_shape'][0]:,} orders and {R['fe_shape'][1]} columns (Data/Feature_Engineered_Dataset.csv). Customer and warehouse history use only information available before each order, and post-outcome fields (delay days, returns, status) are never used as predictors, which prevents leakage. Missing values are structural (first orders have no history) and are median-imputed with explicit missing-flag columns.")]
fd = [['Group', 'Feature', 'Why created', 'Business value', 'ML value'],
 ['Time', 'IsWeekend, IsMonthEnd, DaysSinceStart, Month_sin/cos', 'Calendar cycles and trend', 'Roster and peak planning', 'Seasonality without leakage; time-split index'],
 ['Basket', 'value_per_unit, log_order_value, units_per_line, is_multi_category, is_high_value, is_bulk', 'Describe order size, price level and pick complexity', 'Spot premium or complex orders', 'Scale-free, less skewed predictors'],
 ['Customer', 'cust_prior_orders, cust_prior_value, cust_avg_prior_value, days_since_prev_order, is_repeat_customer, value_vs_cust_avg', 'RFM-style history before the order', 'Loyalty, churn warning, lifetime value', 'Behaviour signals with no look-ahead'],
 ['Warehouse', 'wh_staff, wh_total_stock, wh_low_stock_share, wh_workload_per_emp, wh_prior_delay_rate, wh_orders_last7d, wh_load_index', 'Capacity proxies and rolling performance', 'Staffing and replenishment monitoring', 'Static and dynamic site-load signals'],
 ['Route', 'SameCityDispatch, route', 'Local vs cross-city fulfilment', 'Lane reporting', 'Distance proxy (no coordinates available)'],
 ['Encoding', 'one-hot city and category columns, Status_code', 'Models cannot use raw text', 'None directly', 'One-hot avoids false ordering; few levels']]
st += [table(fd, [1.7*cm, 4.6*cm, 3.5*cm, 3.2*cm, 3.7*cm]), Spacer(1, 6), para(f"Screening: only {R['feature_screen_nsig']} of 24 candidate features reached p < 0.05 against a target (about 2 expected by chance), and the strongest rank correlation was {R['feature_screen_max']:.3f}. Baseline models (logistic regression, random forest, gradient boosting; train on first 70% of dates, test on last 30%) gave test AUC of {R['best_auc_delay']:.2f} for delay and {R['best_auc_return']:.2f} for returns, i.e. no predictive power. The honest conclusion is that the engineered features are sound and ready, but the source data lacks the drivers of delay.")]
st += [fig('24'), PageBreak()]
# 7 Segmentation
st += [Paragraph('7. Segmentation Results', S['h1']), para(f"Customers were segmented by log-scaled recency, frequency and monetary value using K-Means with k = 4 (silhouette {R['silhouette']['4']:.2f}, modest, so segments are planning buckets rather than sharp groups). {R['rfm_no_value_customers']} customers whose confirmed orders carry no line-item value were excluded and reported separately. Products were classed A/B/C by revenue, and warehouses were clustered on workload, delay, stock risk and returns (silhouette {R['wh_silhouette']:.2f})."), fig('25', 10), fig('26', 15)]
rows = [['Segment', 'Customers', 'Avg recency (days)', 'Orders', 'Avg value', '% revenue']] + [[v['segment'], f"{v['customers']:.0f}", f"{v['recency']:.0f}", f"{v['frequency']:.1f}", f"{v['monetary']:,.0f}", f"{v['share_of_confirmed_revenue']*100:.0f}%"] for v in sorted(R['segments'].values(), key=lambda v: -v['share_of_confirmed_revenue'])]
st += [table(rows, [5.5*cm, 2*cm, 2.8*cm, 1.8*cm, 2.3*cm, 2*cm]), Spacer(1, 8), fig('27', 12), fig('28', 13), fig('29', 15), para('Supplier segmentation is not possible because no supplier key exists on any operational table. The warehouse Reliability Score (z-score composite of delay, return rate and low-stock positions) is the best available substitute.'), PageBreak()]
# 8 insights
st += [Paragraph('8. Business Insights', S['h1'])]
for q, a in R['answers'].items(): st += [Paragraph(esc(q), S['h2']), para(a)]
st.append(PageBreak())
# 9 recs
st += [Paragraph('9. Recommendations', S['h1']), para('Fifteen strategic recommendations, each tied to evidence in this report.')]
rows = [['#', 'Recommendation', 'Evidence', 'Priority', 'Expected impact']] + [[str(r['index']), r['recommendation'], r['evidence'], r['priority'], r['expected_impact']] for r in R['recommendations']]
st += [table(rows, [.8*cm, 5.2*cm, 5*cm, 1.4*cm, 4.4*cm]), PageBreak()]
# 10 conclusion
st += [Paragraph('10. Conclusion', S['h1']), para(f"SwiftLog has a stable but flat order book and a serious, network-wide delivery problem: {R['delay_rate']*100:.0f}% of shipments are late, only {R['delivered_rate']*100:.0f}% of orders reach Delivered status, and the records themselves are internally inconsistent. The analysis found no warehouse, region, product or customer group that explains the delays, which means the cause lies in factors that are not recorded today."),
 para('The most valuable next step is therefore data, not modelling: capture supplier, carrier, promised and actual dates, capacity and controlled return reasons, and reconcile order status. With those in place, delay-risk scoring, demand forecasting and churn prediction become feasible. The feature pipeline and cleaning functions built here can be re-run unchanged on richer data.'),
 Paragraph('Limitations', S['h2'])] + bullets(R['limitations']) + [Paragraph('Future ML opportunities', S['h2'])] + bullets(['Delivery-delay risk scoring once carrier, route distance and promised dates are captured', 'Demand and revenue forecasting by category and city (needs a longer history than 8 months)', 'Customer churn / win-back propensity from the RFM features built here', 'Return-risk scoring once product-level defect and picker data exist', 'Anomaly detection on order status vs shipment events to catch process errors early'])
SimpleDocTemplate(P + 'Reports/Technical_Report.pdf', pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=1.8*cm, bottomMargin=1.8*cm, title='SwiftLog Technical Report', author='AnalystLab Africa Consulting').build(st, onFirstPage=footer, onLaterPages=footer)
# =================== EXEC SUMMARY ===================
e = [Paragraph('SwiftLog Logistics — Executive Summary', S['ti']), para('Supply chain intelligence review · orders 1 Jan – 8 Sep 2026 · AnalystLab Africa Consulting', 'sub'), Spacer(1, 8)]
k = [['Confirmed order value', 'Delivered orders', 'Shipments delayed (3+ days)', 'Orders with a return'], [f(R['rev_confirmed']), f"{R['delivered_rate']*100:.1f}%", f"{R['delay_rate']*100:.1f}%", f"{R['return_rate']*100:.1f}%"]]
kt = Table([[Paragraph(esc(c), S['cellb'] if i == 0 else ParagraphStyle('kv', parent=S['h1'], fontSize=17, textColor=INK, spaceBefore=0, spaceAfter=0)) for c in r] for i, r in enumerate(k)], colWidths=[4.25*cm]*4); kt.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), BLUE), ('BACKGROUND', (0, 1), (-1, 1), LIGHT), ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5), ('BOX', (0, 0), (-1, -1), .4, colors.HexColor('#d9dee8'))]))
e += [kt, Spacer(1, 8), Paragraph('What we found', S['h2'])] + bullets([
 f"Only one order in three is delivered. {R['cancel_rate']*100:.0f}% are cancelled ({f(R['cancelled_value'])} of value) and {R['pending_rate']*100:.0f}% are pending. Closing this gap is a bigger prize than winning new demand.",
 f"Late delivery is the norm, not the exception: {R['delay_rate']*100:.0f}% of shipments arrive 3 or more days late. No warehouse, city, product or order type is reliably worse, so the cause is network-wide.",
 f"Records disagree with each other: {R['ship_on_cancelled']:,} shipments are logged against cancelled orders, so delivery KPIs need reconciling before they are trusted.",
 f"Revenue is flat and evenly spread. The top 20% of customers give {R['cust_top20_share']*100:.0f}% of revenue; a third of revenue sits with high-value customers who have not ordered for about {max(v['recency'] for v in R['segments'].values() if v['segment'].startswith('At-risk')):.0f} days.",
 f"Staffing is uneven: shipments per employee range from 20 to 38 across warehouses, and low stock is concentrated in W14, W06, W13 and W15.",
 'Supplier performance cannot be measured: no table links suppliers to products or shipments.',
 f"Predicting delays is not possible with today's data (best AUC {R['best_auc_delay']:.2f}, chance is 0.50)."]) + [Paragraph('What we recommend', S['h2'])] + bullets([
 'Reconcile order status with shipment and return events (high priority).', 'Launch a network-wide on-time delivery programme covering carrier SLAs, dispatch cut-offs and promise dates.', 'Run a funnel review of cancelled and pending orders; recovering 10% of cancelled value is worth about ' + f(R['cancelled_value'] * .1) + '.',
 'Rebalance staff toward W14, W15 and W12 and replenish stock at the low-stock hotspots.', 'Start a win-back programme for lapsing high-value customers.', 'Capture supplier, carrier, promised and actual dates, and warehouse capacity, then build delay-risk and demand models.']) + [
 Spacer(1, 6), Image(P + 'Visualizations/Dashboards/Dashboard_1_Executive_KPIs.png', 17*cm, 17*cm * PI.open(P + 'Visualizations/Dashboards/Dashboard_1_Executive_KPIs.png').size[1] / PI.open(P + 'Visualizations/Dashboards/Dashboard_1_Executive_KPIs.png').size[0]), Paragraph('Data notes: currency units are as supplied; September 2026 is a partial month (to 8 Sep); full methods and all tests are in the Technical Report.', S['sm'])]
SimpleDocTemplate(P + 'Reports/Executive_Summary.pdf', pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=1.6*cm, bottomMargin=1.6*cm, title='SwiftLog Executive Summary', author='AnalystLab Africa Consulting').build(e, onFirstPage=footer, onLaterPages=footer)
print('ok')
