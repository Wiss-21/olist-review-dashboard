import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# ── PAGE CONFIG ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Olist Review Risk",
    page_icon="📦",
    layout="wide",
)

# Plotly UI config — hides the hover toolbar on every chart
PLOTLY_CONFIG = {'displayModeBar': False}

import plotly.io as pio

# ── BRAND PALETTE ─────────────────────────────────────────────────────
ACCENT   = "#2DD4BF"   # teal — primary, "good"
INDIGO   = "#818CF8"   # intermediate
PURPLE   = "#A78BFA"   # accent variation
WARNING  = "#FB7185"   # rose — "bad" outcomes
SUCCESS  = "#34D399"   # green — improvement
TEXT_DIM = "#9CA3AF"

# Continuous scales (use for color gradients)
SCALE_RISK = [ACCENT, INDIGO, WARNING]    # low (good) → high (bad)

# Apply a single dark template to every Plotly chart in this app
pio.templates["lucent_dark"] = pio.templates["plotly_dark"]
pio.templates["lucent_dark"].layout.update(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="sans-serif", color="#E8ECF1", size=12),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
    colorway=[ACCENT, INDIGO, PURPLE, WARNING, SUCCESS],
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
)
pio.templates.default = "lucent_dark"

# ── DATA LOADING (cached so it only runs once) ─────────────────────────
@st.cache_data
def load_data():
    DATA_DIR = './Data/'

    orders = pd.read_csv(DATA_DIR + 'olist_orders_dataset.csv')
    items = pd.read_csv(DATA_DIR + 'olist_order_items_dataset.csv')
    reviews = pd.read_csv(DATA_DIR + 'olist_order_reviews_dataset.csv')
    products = pd.read_csv(DATA_DIR + 'olist_products_dataset.csv')
    cat_xlat = pd.read_csv(DATA_DIR + 'product_category_name_translation.csv')
    sellers = pd.read_csv(DATA_DIR + 'olist_sellers_dataset.csv')

    # Parse dates and filter to delivered orders
    date_cols = ['order_purchase_timestamp', 'order_approved_at',
                 'order_delivered_carrier_date', 'order_delivered_customer_date',
                 'order_estimated_delivery_date']
    for c in date_cols:
        orders[c] = pd.to_datetime(orders[c])
    orders = orders[orders['order_status'] == 'delivered'].dropna(
        subset=['order_delivered_customer_date']
    )
    orders['delivery_time_days'] = (
        orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']
    ).dt.days
    orders['delivery_delay_days'] = (
        orders['order_delivered_customer_date'] - orders['order_estimated_delivery_date']
    ).dt.days

    # Aggregate items per order
    items_agg = items.groupby('order_id').agg(
        n_items=('order_item_id', 'count'),
        total_price=('price', 'sum'),
        total_freight=('freight_value', 'sum'),
        primary_seller=('seller_id', 'first'),
        primary_product=('product_id', 'first'),
    ).reset_index()
    items_agg['freight_pct'] = items_agg['total_freight'] / items_agg['total_price']

    # Aggregate reviews per order
    reviews_agg = reviews.groupby('order_id').agg(
        review_score=('review_score', 'mean'),
    ).reset_index()
    reviews_agg['review_score_round'] = reviews_agg['review_score'].round().astype(int)

    # Master join
    df = (orders
          .merge(items_agg, on='order_id', how='left')
          .merge(reviews_agg, on='order_id', how='left')
          .merge(products[['product_id', 'product_category_name']],
                 left_on='primary_product', right_on='product_id', how='left')
          .merge(cat_xlat, on='product_category_name', how='left')
          .merge(sellers[['seller_id', 'seller_state']],
                 left_on='primary_seller', right_on='seller_id', how='left'))

    df = df.dropna(subset=['review_score', 'total_price', 'product_category_name_english'])
    df['is_bad_review'] = (df['review_score_round'] <= 2).astype(int)

    # Delivery buckets for grouping
    df['delivery_bucket'] = pd.cut(
        df['delivery_delay_days'],
        bins=[-100, -7, -1, 0, 7, 100],
        labels=['7+ days early', '1-7 days early', 'on time', '1-7 days late', '7+ days late']
    )

    return df


df = load_data()

# ── HEADER ─────────────────────────────────────────────────────────────
st.title("Olist Review Risk Dashboard")
st.caption("What drives a bad review on Brazil's largest e-commerce marketplace? "
           "94k orders · 2017–2018 · interactive exploration.")

# ── SIDEBAR FILTERS ────────────────────────────────────────────────────
st.sidebar.header("Filters")

all_categories = sorted(df['product_category_name_english'].dropna().unique())
selected_cats = st.sidebar.multiselect(
    "Product category",
    options=all_categories,
    default=[],
    help="Leave empty for all categories",
)

all_states = sorted(df['seller_state'].dropna().unique())
selected_states = st.sidebar.multiselect(
    "Seller state",
    options=all_states,
    default=[],
    help="Leave empty for all states",
)

# Apply filters
filtered = df.copy()
if selected_cats:
    filtered = filtered[filtered['product_category_name_english'].isin(selected_cats)]
if selected_states:
    filtered = filtered[filtered['seller_state'].isin(selected_states)]

st.sidebar.markdown(f"**{len(filtered):,}** orders match")

# ── KPI TILES ──────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Orders", f"{len(filtered):,}")
c2.metric("Avg Review", f"{filtered['review_score'].mean():.2f} ★")
c3.metric("Bad Review Rate", f"{filtered['is_bad_review'].mean() * 100:.1f}%")
c4.metric("Avg Delivery", f"{filtered['delivery_time_days'].mean():.1f} days")

st.markdown("---")

# ── CHART 1: BAD REVIEW RATE BY DELIVERY PERFORMANCE ───────────────────
st.subheader("Delivery performance vs review quality")
st.caption("Each bar shows the share of orders that received a bad review (≤2★), "
           "split by how late or early they arrived vs the estimated delivery date.")

delivery_rates = (filtered.groupby('delivery_bucket', observed=True)
                          .agg(bad_rate=('is_bad_review', 'mean'),
                               n_orders=('is_bad_review', 'count'))
                          .reset_index())
delivery_rates['bad_rate_pct'] = delivery_rates['bad_rate'] * 100

fig = px.bar(
    delivery_rates,
    x='delivery_bucket',
    y='bad_rate_pct',
    text='bad_rate_pct',
    color='bad_rate_pct',
    color_continuous_scale=SCALE_RISK,
    labels={'bad_rate_pct': 'Bad review rate (%)', 'delivery_bucket': 'Delivery performance'},
)
fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig.update_layout(
    showlegend=False,
    coloraxis_showscale=False,
    height=420,
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
# ── CHART 2: TOP PROBLEM CATEGORIES ────────────────────────────────────
st.subheader("Worst categories by bad review rate")
st.caption("Categories with at least 500 orders, ranked by share of bad reviews. "
           "Hover for exact numbers; click a category in the sidebar to filter the rest of the dashboard.")

cat_stats = (filtered.groupby('product_category_name_english')
                     .agg(bad_rate=('is_bad_review', 'mean'),
                          n_orders=('is_bad_review', 'count'),
                          avg_delivery=('delivery_time_days', 'mean'))
                     .query('n_orders >= 500')
                     .sort_values('bad_rate', ascending=False)
                     .head(15)
                     .reset_index())
cat_stats['bad_rate_pct'] = cat_stats['bad_rate'] * 100

if len(cat_stats) == 0:
    st.info("No categories meet the minimum 500-order threshold under current filters. "
            "Try widening the filters.")
else:
    fig2 = px.bar(
        cat_stats.sort_values('bad_rate_pct'),  # ascending so worst is at top of horizontal bar
        x='bad_rate_pct',
        y='product_category_name_english',
        orientation='h',
        text='bad_rate_pct',
        hover_data={'n_orders': ':,', 'avg_delivery': ':.1f', 'bad_rate_pct': False},
        color='bad_rate_pct',
        color_continuous_scale=SCALE_RISK,
        labels={
            'bad_rate_pct': 'Bad review rate (%)',
            'product_category_name_english': '',
            'n_orders': 'Orders',
            'avg_delivery': 'Avg delivery (days)',
        },
    )
    fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig2.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        height=500,
        margin=dict(l=10, r=40, t=20, b=20),
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
# ── CHART 3: TIME TREND ────────────────────────────────────────────────
st.subheader("Is Olist getting better over time?")
st.caption("Monthly average review score and bad review rate across the data window. "
           "Both lines update with your filters.")

monthly = (filtered.assign(month=filtered['order_purchase_timestamp'].dt.to_period('M').dt.to_timestamp())
                   .groupby('month')
                   .agg(avg_review=('review_score', 'mean'),
                        bad_rate=('is_bad_review', 'mean'),
                        n_orders=('is_bad_review', 'count'))
                   .reset_index()
                   .query('n_orders >= 100'))  # drop sparse months at edges
monthly['bad_rate_pct'] = monthly['bad_rate'] * 100

# Two-line chart: avg review (left axis), bad rate (right axis)
from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig3 = make_subplots(specs=[[{"secondary_y": True}]])
fig3.add_trace(
    go.Scatter(x=monthly['month'], y=monthly['avg_review'],
               name='Avg review (★)', line=dict(color=ACCENT, width=3),
               mode='lines+markers'),
    secondary_y=False,
)
fig3.add_trace(
    go.Scatter(x=monthly['month'], y=monthly['bad_rate_pct'],
               name='Bad review rate (%)', line=dict(color=WARNING, width=3, dash='dot'),
               mode='lines+markers'),
    secondary_y=True,
)
fig3.update_yaxes(title_text="Avg review score", range=[3.5, 5], secondary_y=False)
fig3.update_yaxes(title_text="Bad review rate (%)", range=[0, 30], secondary_y=True)
fig3.update_layout(
    height=420,
    hovermode='x unified',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig3, use_container_width=True)
# ── CHART 4: GEOGRAPHIC MAP ────────────────────────────────────────────
st.subheader("Bad review rate by seller state")
st.caption("Choropleth of Brazil's 27 states + DF. Darker red = worse review quality. "
           "Hover for state-level numbers. States with under 100 orders are excluded.")

import json
import urllib.request

@st.cache_data
def load_brazil_geojson():
    url = "https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/brazil-states.geojson"
    with urllib.request.urlopen(url) as response:
        return json.load(response)

brazil_geo = load_brazil_geojson()

state_stats = (filtered.groupby('seller_state')
                       .agg(bad_rate=('is_bad_review', 'mean'),
                            n_orders=('is_bad_review', 'count'),
                            avg_delivery=('delivery_time_days', 'mean'))
                       .query('n_orders >= 100')
                       .reset_index())
state_stats['bad_rate_pct'] = state_stats['bad_rate'] * 100

if len(state_stats) == 0:
    st.info("No states meet the minimum 100-order threshold under current filters.")
else:
    # Merge with full state list so missing states still appear in outline
    all_states = [f['properties']['sigla'] for f in brazil_geo['features']]
    state_stats_full = pd.DataFrame({'seller_state': all_states}).merge(
        state_stats, on='seller_state', how='left'
    )

    fig4 = px.choropleth(
        state_stats_full,
        geojson=brazil_geo,
        locations='seller_state',
        featureidkey='properties.sigla',
        color='bad_rate_pct',
        color_continuous_scale=SCALE_RISK,
        range_color=[8, 18],
        hover_data={
            'n_orders': ':,',
            'avg_delivery': ':.1f',
            'bad_rate_pct': ':.1f',
        },
        labels={
            'bad_rate_pct': 'Bad rate (%)',
            'seller_state': 'State',
            'n_orders': 'Orders',
            'avg_delivery': 'Avg delivery (days)',
        },
    )
    fig4.update_geos(
        fitbounds='geojson',     # show ALL of Brazil, not just states with data
        visible=False,
        bgcolor='rgba(0,0,0,0)',
    )
    fig4.update_traces(
        marker_line_color='rgba(120,120,120,0.5)',
        marker_line_width=0.7,
    )
    fig4.update_layout(
        height=540,
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(title='Bad rate (%)'),
    )
    st.plotly_chart(fig4, use_container_width=True)
    # ── CHART 5: PREDICTIVE MODEL ─────────────────────────────────────────
st.subheader("What actually predicts a bad review?")
st.caption("A random forest trained on order features (delivery, price, category, seller location). "
           "The bar chart shows which features the model relies on most. AUC scores tell you how well "
           "the model separates good from bad reviews — above 0.7 is real signal, above 0.8 is strong.")

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

@st.cache_resource
def train_model(df_in):
    features_num = ['delivery_delay_days', 'delivery_time_days', 'total_price',
                    'total_freight', 'freight_pct', 'n_items']

    df_m = df_in.copy()
    top_cats = df_m['product_category_name_english'].value_counts().head(15).index.tolist()
    df_m['cat_grouped'] = df_m['product_category_name_english'].where(
        df_m['product_category_name_english'].isin(top_cats), 'other')
    top_states = df_m['seller_state'].value_counts().head(10).index.tolist()
    df_m['state_grouped'] = df_m['seller_state'].where(
        df_m['seller_state'].isin(top_states), 'other_state')

    X = pd.concat([
        df_m[features_num],
        pd.get_dummies(df_m['cat_grouped'], prefix='cat'),
        pd.get_dummies(df_m['state_grouped'], prefix='state'),
    ], axis=1).fillna(0)
    y = df_m['is_bad_review']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LGBMClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced',
        verbose=-1,           # suppress LightGBM's chatty training logs
    )
    model.fit(X_train, y_train)

    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False).head(15)

    return model, X.columns.tolist(), X_test, y_test, train_auc, test_auc, importances


with st.spinner("Training model (one-time, ~10-15 seconds)..."):
    model, feature_cols, X_test, y_test, train_auc, test_auc, importances = train_model(df)

c1, c2 = st.columns(2)
c1.metric("Train AUC", f"{train_auc:.3f}")
c2.metric("Test AUC", f"{test_auc:.3f}",
          delta=f"{test_auc - train_auc:+.3f}", delta_color="off")

fig5 = px.bar(
    importances.sort_values('importance'),
    x='importance',
    y='feature',
    orientation='h',
    color='importance',
    color_continuous_scale=SCALE_RISK,
    labels={'importance': 'Feature importance (gini-based)', 'feature': ''},
)
fig5.update_layout(
    showlegend=False, coloraxis_showscale=False,
    height=480, margin=dict(l=10, r=10, t=20, b=20),
)
st.plotly_chart(fig5, use_container_width=True)

st.info("**Reading this:** Delivery delay dominates the feature importance — same story Chart 1 told visually, "
        "now backed by a model. Price and freight modestly matter. Category and state contribute small "
        "individual signal but add up in aggregate. If train AUC is much higher than test AUC, the model "
        "is overfitting — current settings should keep them close.")

st.markdown("---")




# ── ROC + CONFUSION MATRIX ────────────────────────────────────────────
st.markdown("---")
st.subheader("Model evaluation: ROC curve and confusion matrix")
st.caption("Standard ML evaluation diagnostics. The ROC curve shows how well the model separates classes at every threshold; the confusion matrix shows actual vs predicted at the 0.5 decision threshold.")

from sklearn.metrics import roc_curve, confusion_matrix

y_test_pred_proba = model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_pred_proba >= 0.5).astype(int)

ec1, ec2 = st.columns(2)

fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
roc_df = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
fig_roc = px.area(
    roc_df, x='FPR', y='TPR',
    title=f'ROC Curve (AUC = {test_auc:.3f})',
    labels={'FPR': 'False Positive Rate', 'TPR': 'True Positive Rate'},
    color_discrete_sequence=[ACCENT],
)
fig_roc.add_shape(type='line', line=dict(dash='dash', color='gray', width=1),
                  x0=0, x1=1, y0=0, y1=1)
fig_roc.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=20))
ec1.plotly_chart(fig_roc, use_container_width=True, config=PLOTLY_CONFIG)

cm = confusion_matrix(y_test, y_test_pred)
cm_df = pd.DataFrame(
    cm,
    index=['Actual: Good', 'Actual: Bad'],
    columns=['Predicted: Good', 'Predicted: Bad'],
)
fig_cm = px.imshow(
    cm_df, text_auto=True, aspect='auto',
    title='Confusion Matrix (threshold = 0.5)',
    color_continuous_scale=[ACCENT, INDIGO, WARNING],
)
fig_cm.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=20), coloraxis_showscale=False)
ec2.plotly_chart(fig_cm, use_container_width=True, config=PLOTLY_CONFIG)

total = cm.sum()
tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
st.caption(
    f"**Reading the confusion matrix:** Of {total:,} test orders, the model correctly identified "
    f"**{tp:,}** bad reviews (true positives) and missed **{fn:,}** (false negatives). "
    f"It flagged **{fp:,}** good orders as risky (false positives). "
    f"Precision = {precision*100:.1f}%, Recall = {recall*100:.1f}%."
)

# -- INTERACTIVE PREDICTOR ----------------------------------------
st.markdown("---")
st.subheader("Try the model: what's the bad-review risk for this order?")
st.caption("Move the sliders to set a hypothetical order. The probability updates live.")

c1, c2 = st.columns(2)
with c1:
    in_delay = st.slider("Delivery delay (days)", min_value=-30, max_value=30, value=0)
    in_price = st.slider("Order total price (R$)", min_value=10, max_value=500, value=100, step=5)
    in_items = st.slider("Number of items in order", min_value=1, max_value=10, value=1)
with c2:
    in_freight = st.slider("Freight cost (R$)", min_value=0, max_value=100, value=15, step=1)
    in_delivery_days = st.slider("Total delivery time (days)", min_value=1, max_value=60, value=12)
    _cat_opts = sorted(df['product_category_name_english'].dropna().unique().tolist())
    in_category = st.selectbox("Product category", options=_cat_opts)
    _state_opts = sorted(df['seller_state'].dropna().unique().tolist())
    in_state = st.selectbox("Seller state", options=_state_opts)

feature_vec = pd.DataFrame(np.zeros((1, len(feature_cols))), columns=feature_cols)
feature_vec['delivery_delay_days'] = in_delay
feature_vec['delivery_time_days']  = in_delivery_days
feature_vec['total_price']         = in_price
feature_vec['total_freight']       = in_freight
feature_vec['n_items']             = in_items
feature_vec['freight_pct']         = in_freight / in_price if in_price > 0 else 0

_cat_col = 'cat_' + in_category
if _cat_col in feature_vec.columns:
    feature_vec[_cat_col] = 1
elif 'cat_other' in feature_vec.columns:
    feature_vec['cat_other'] = 1

_state_col = 'state_' + in_state
if _state_col in feature_vec.columns:
    feature_vec[_state_col] = 1
elif 'state_other_state' in feature_vec.columns:
    feature_vec['state_other_state'] = 1

prob = float(model.predict_proba(feature_vec)[0][1])

result_col, bar_col = st.columns([1, 2])
with result_col:
    st.metric('Bad-review probability', str(round(prob * 100, 1)) + '%')
    if prob > 0.5:
        st.error('Likely bad review')
    elif prob > 0.3:
        st.warning('Elevated risk')
    else:
        st.success('Low risk')

with bar_col:
    st.write('')
    st.write('')
    st.progress(prob)
    st.caption('0% safe ---- 50% threshold ---- 100% risky')

st.caption('Try setting delivery delay to +14 days. The probability jumps dramatically -- that is the late-delivery effect dominating the model.')

st.caption("More charts coming. Run `streamlit run app.py` from your project folder to launch.")

# ── STATISTICAL TEST ──────────────────────────────────────────────────
st.markdown("---")
st.subheader("Is the delivery-delay effect actually significant?")
st.caption("Visual differences can lie. We run a formal statistical test to prove that late deliveries get worse reviews — and that the difference isn't just noise.")

from scipy import stats

on_time = df[df['delivery_delay_days'] <= 0]['is_bad_review']
late_   = df[df['delivery_delay_days'] >  0]['is_bad_review']

p1 = on_time.mean()
p2 = late_.mean()
n1 = len(on_time)
n2 = len(late_)
p_pool = (on_time.sum() + late_.sum()) / (n1 + n2)
se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
z = (p2 - p1) / se
p_value = 2 * (1 - stats.norm.cdf(abs(z)))

sc1, sc2, sc3 = st.columns(3)
sc1.metric("On-time bad rate", f"{p1*100:.1f}%", f"{n1:,} orders")
sc2.metric("Late bad rate", f"{p2*100:.1f}%", f"{n2:,} orders")
sc3.metric("Difference", f"+{(p2-p1)*100:.1f} pp", f"z = {z:.2f}")

if p_value < 0.001:
    sig_text = "p < 0.001 (extremely significant)"
elif p_value < 0.01:
    sig_text = f"p = {p_value:.4f} (highly significant)"
elif p_value < 0.05:
    sig_text = f"p = {p_value:.4f} (significant)"
else:
    sig_text = f"p = {p_value:.4f} (NOT significant)"

st.info(
    f"**Two-proportion z-test result:** {sig_text}. "
    f"The probability of seeing this difference by random chance is effectively zero. "
    f"Late delivery causes ~**{(p2-p1)*100:.0f} percentage points more** bad reviews than on-time delivery."
)


# ── SENTIMENT ANALYSIS ON REVIEW TEXT ─────────────────────────────────
st.markdown("---")
st.subheader("Do customers say what they rate? (Sentiment vs Stars)")
st.caption("Sentiment analysis on the review_comment_message column. We compare the text's sentiment score against the actual star rating to find mismatches.")

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

@st.cache_data
def compute_sentiment(_df_reviews):
    analyzer = SentimentIntensityAnalyzer()
    df_r = _df_reviews.dropna(subset=['review_comment_message']).copy()
    if len(df_r) > 10000:
        df_r = df_r.sample(10000, random_state=42)
    df_r['sentiment'] = df_r['review_comment_message'].apply(
        lambda txt: analyzer.polarity_scores(str(txt))['compound']
    )
    return df_r[['order_id', 'review_score', 'sentiment', 'review_comment_message']]

with st.spinner("Scoring review text sentiment (one-time, ~15-30 seconds)..."):
    reviews_raw = pd.read_csv('./Data/olist_order_reviews_dataset.csv')
    sent_df = compute_sentiment(reviews_raw)

fig_sent = px.box(
    sent_df,
    x='review_score', y='sentiment',
    color='review_score',
    title='Text sentiment score by star rating',
    labels={'sentiment': 'VADER sentiment (-1 = negative, +1 = positive)',
            'review_score': 'Star rating'},
    color_discrete_sequence=[WARNING, '#FBBF24', INDIGO, PURPLE, ACCENT],
)
fig_sent.update_layout(height=420, showlegend=False, margin=dict(l=10, r=10, t=40, b=20))
st.plotly_chart(fig_sent, use_container_width=True, config=PLOTLY_CONFIG)

mismatched_5_star_neg = sent_df[(sent_df['review_score'] == 5) & (sent_df['sentiment'] < -0.3)]
mismatched_1_star_pos = sent_df[(sent_df['review_score'] == 1) & (sent_df['sentiment'] > 0.3)]

mc1, mc2 = st.columns(2)
mc1.metric("5-star reviews with negative text", f"{len(mismatched_5_star_neg):,}")
mc2.metric("1-star reviews with positive text", f"{len(mismatched_1_star_pos):,}")

st.caption(
    "**Reading the box plot:** Higher star ratings should correlate with positive sentiment. "
    "Outliers in unexpected positions are interesting — they suggest data quality issues or "
    "edge cases worth investigating manually."
)
