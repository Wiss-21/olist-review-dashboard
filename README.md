# Olist Review Risk Dashboard

A live interactive Streamlit app that predicts which Olist (Brazilian e-commerce) orders are likely to receive a bad review — and lets users explore the data behind it.

**Live demo:** **[olist-review-wissam.streamlit.app](https://olist-review-wissam.streamlit.app)**

## Headline findings

- **Delivery delay dominates everything.** Orders arriving 7+ days late get a **79.8%** bad-review rate vs **8.9%** for orders delivered 7+ days early. Confirmed visually, by hypothesis test (two-proportion z-test, p < 0.001), and by the trained LightGBM model.
- **Category matters less than expected.** The worst categories cap at ~16% bad-review rate; the best ones sit at ~11%. The variance in delivery (9% → 80%) dwarfs the variance in category (~5pp).
- **Olist recovered from a Q1 2018 quality crisis.** Bad-review rate spiked to ~21% in March 2018 before recovering to ~10% by mid-2018.
- **Sentiment ≠ stars.** VADER sentiment analysis on review text reveals hundreds of mismatches — 5-star reviews with negative text and 1-star reviews with positive text. Real data quality signal worth surfacing for product teams.

## What is in the dashboard

Nine interactive sections, all filterable by category and seller state:

1. **KPI tiles** — Orders, avg review, bad-review rate, avg delivery time
2. **Delivery performance** vs review quality (Plotly bar chart)
3. **Worst categories** by bad-review rate (Plotly horizontal bar)
4. **Time trend** — Avg review + bad-rate over 2017-2018 (dual-axis line chart)
5. **Brazil choropleth** — Bad-review rate by seller state (Plotly geo map)
6. **LightGBM classifier** with feature importance (gradient boosting, ~0.78 test AUC)
7. **ROC curve + confusion matrix** — Model evaluation diagnostics
8. **Interactive predictor** — Sliders for order features, live bad-review probability with green-to-red gauge
9. **Two-proportion z-test** — Formal hypothesis test for the delivery-delay effect
10. **VADER sentiment analysis** — Box plot of text sentiment by star rating, plus mismatch counters

## Stack

- **Python** · pandas, numpy
- **Streamlit** for the dashboard framework + deployment
- **Plotly** for all interactive charts (bar, line, choropleth, box)
- **LightGBM** for the classifier (gradient boosting)
- **scikit-learn** for train/test split, ROC, confusion matrix
- **scipy** for the two-proportion z-test
- **VADER (vaderSentiment)** for NLP sentiment scoring

## Reproducing the analysis locally

1. Download the Olist dataset from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (free, requires Kaggle account).
2. Place all 9 CSV files in a folder named `Data/` at the project root. The app uses 6 of them: `olist_orders_dataset.csv`, `olist_order_items_dataset.csv`, `olist_order_reviews_dataset.csv`, `olist_products_dataset.csv`, `olist_sellers_dataset.csv`, `product_category_name_translation.csv`.
3. Install dependencies: `pip install -r requirements.txt`
4. Launch the app: `streamlit run app.py`
5. Browser opens at `localhost:8501`.

## Project structureolist-review-dashboard/

├── app.py                  # Streamlit application

├── requirements.txt        # Python dependencies

├── README.md               # This file

├── .streamlit/

│   └── config.toml         # Dark theme config

├── Data/                   # CSV files (not committed — download from Kaggle)

└── Notebooks/

└── 01_exploration.ipynb  # Initial data explorationThe CSV data files (~120 MB) are not committed to this repo — see the reproduction steps above for the download.

## About

Built by Wissam Ezzedine as part of my data portfolio.

- **Portfolio:** [wissam-ezzedine.netlify.app](https://wissam-ezzedine.netlify.app)
- **LinkedIn:** [linkedin.com/in/wissam-ezzedine](https://www.linkedin.com/in/wissam-ezzedine)
- **Contact:** wissamezzedine123@outlook.com

