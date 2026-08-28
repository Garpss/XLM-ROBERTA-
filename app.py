from __future__ import annotations

import io
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model import (
    LABEL_COLORS,
    LABEL_ORDER,
    ModelMetadata,
    SentimentModelService,
    aggregate_by_product,
    apply_bayesian_score,
    evaluate_ranking,
    load_hybrid_metrics,
    parse_project_metadata,
    prediction_history_row,
    rank_products,
    raw_sentiment_average_rank,
    resolve_model_source,
)

st.set_page_config(
    page_title="Harnessing Sentiment Analysis for Smarter Recommendations",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
def _inject_theme() -> None:
    st.markdown(
        """
        <style>
            .stApp { background-color: #0E1117; color: #E8EDF3; }
            .hero {
                padding: 1.6rem 1.9rem;
                border-radius: 1rem;
                background: linear-gradient(135deg, #1B2431 0%, #141A22 100%);
                border: 1px solid #2A3440;
                margin-bottom: 1.1rem;
            }
            .hero h1 { margin: 0; font-size: 1.9rem; letter-spacing: 0.2px; }
            .subtle { color: #A8B3C1; }
            .pill {
                display: inline-block;
                padding: 0.28rem 0.7rem;
                border-radius: 999px;
                background-color: #222B36;
                border: 1px solid #33404F;
                margin-right: 0.35rem;
                margin-bottom: 0.35rem;
                font-size: 0.8rem;
            }
            .step-card {
                padding: 1rem 1.1rem;
                border-radius: 0.8rem;
                background-color: #161C24;
                border: 1px solid #263140;
                height: 100%;
            }
            .step-card h4 { margin: 0 0 0.35rem 0; color: #E8EDF3; }
            .step-card p { margin: 0; color: #9AA7B5; font-size: 0.86rem; }
            .badge-pos { color: #2ECC71; font-weight: 600; }
            .badge-neg { color: #E74C3C; font-weight: 600; }
            .badge-neu { color: #95A5A6; font-weight: 600; }
            .footer { color: #6B7684; font-size: 0.8rem; margin-top: 2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Model loading (cached once)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading fine-tuned XLM-RoBERTa model...")
def load_service(
    notebook_path: str, model_dir: str
) -> Tuple[ModelMetadata, SentimentModelService, str, bool, str]:
    metadata = parse_project_metadata(notebook_path=notebook_path or None)
    resolved_source, runtime_note = resolve_model_source(
        metadata, explicit_model_dir=model_dir or None
    )
    metadata.model_runtime_note = runtime_note
    service = SentimentModelService(resolved_source)
    schema_ok, schema_message = service.validate_expected_label_schema()
    return metadata, service, resolved_source, schema_ok, schema_message


def _init_session_state() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("analyzed_reviews", None)
    st.session_state.setdefault("aggregated", None)
    st.session_state.setdefault("bayesian_params", None)
    st.session_state.setdefault("product_col", None)
    st.session_state.setdefault("review_pool", None)
    st.session_state.setdefault("picked_review_idx", 0)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def _render_sidebar(
    metadata: ModelMetadata, service: SentimentModelService, model_source: str, schema_ok: bool
) -> None:
    st.sidebar.header("Model Info")
    st.sidebar.markdown(f"**Backbone:** `{metadata.model_name}`")
    st.sidebar.markdown(f"**Number of labels:** `{metadata.num_labels}`")
    st.sidebar.markdown(f"**Runtime label map:** `{service.runtime_label_map()}`")

    if schema_ok:
        st.sidebar.success("Sentiment model loaded.")
    else:
        st.sidebar.error("Sentiment checkpoint NOT loaded (see banner).")

    st.sidebar.divider()
    st.sidebar.subheader("Study Datasets")
    st.sidebar.caption(metadata.dataset_note)
    st.sidebar.markdown(f"- Split ratio: `{metadata.split_ratio}`")

    st.sidebar.divider()
    st.sidebar.subheader("Training Info")
    if metadata.labeled_rows is not None:
        st.sidebar.markdown(f"- Labeled rows: `{metadata.labeled_rows}`")
    if metadata.train_rows is not None and metadata.val_rows is not None:
        st.sidebar.markdown(f"- Train / Val: `{metadata.train_rows}` / `{metadata.val_rows}`")
    if metadata.class_distribution:
        st.sidebar.markdown(f"- Class dist.: `{metadata.class_distribution}`")
    if metadata.training_hyperparameters:
        st.sidebar.markdown(f"- Hyperparams: `{metadata.training_hyperparameters}`")
    if metadata.eval_metrics:
        st.sidebar.markdown("- Reported eval (notebook):")
        for key, value in metadata.eval_metrics.items():
            st.sidebar.markdown(f"    - `{key}`: `{value:.4f}`")

    st.sidebar.divider()
    st.sidebar.subheader("Supported Languages")
    st.sidebar.markdown(", ".join(metadata.supported_languages))
    st.sidebar.caption(metadata.language_support_note)

    st.sidebar.divider()
    st.sidebar.subheader("Authors")
    for author in metadata.authors:
        st.sidebar.markdown(f"- {author}")
    st.sidebar.caption(metadata.institution)

    st.sidebar.divider()
    st.sidebar.caption(f"Model source: `{model_source}`")
    if metadata.notebook_path:
        st.sidebar.caption(f"Metadata source: `{metadata.notebook_path}`")


# --------------------------------------------------------------------------- #
# Reusable visuals
# --------------------------------------------------------------------------- #
def _sentiment_badge(label: str) -> str:
    css = {"positive": "badge-pos", "negative": "badge-neg", "neutral": "badge-neu"}
    return f"<span class='{css.get(label, 'badge-neu')}'>{label.title()}</span>"


def _probability_chart(probabilities: dict) -> go.Figure:
    labels = [lbl for lbl in LABEL_ORDER if lbl in probabilities] or list(probabilities.keys())
    values = [probabilities[label] for label in labels]
    colors = [LABEL_COLORS.get(label, "#7F8C8D") for label in labels]

    figure = go.Figure(
        data=[
            go.Bar(
                x=[label.title() for label in labels],
                y=values,
                marker_color=colors,
                text=[f"{value:.3f}" for value in values],
                textposition="outside",
            )
        ]
    )
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Probability",
        xaxis_title="Sentiment Class",
        yaxis=dict(range=[0, 1]),
        margin=dict(l=20, r=20, t=30, b=20),
        height=360,
    )
    return figure


def _recommendation_chart(ranked: pd.DataFrame, product_col: str) -> go.Figure:
    plot_df = ranked.iloc[::-1]
    figure = go.Figure(
        go.Bar(
            x=plot_df["bayesian_score"],
            y=plot_df[product_col].astype(str).str.slice(0, 45),
            orientation="h",
            marker=dict(
                color=plot_df["bayesian_score"],
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="Score"),
            ),
            text=[f"{v:.3f}" for v in plot_df["bayesian_score"]],
            textposition="outside",
        )
    )
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Bayesian Weighted Score",
        yaxis_title="",
        margin=dict(l=20, r=20, t=30, b=20),
        height=460,
    )
    return figure


def _confusion_matrix_chart(cm, labels) -> go.Figure:
    z = cm.tolist()
    figure = go.Figure(
        data=go.Heatmap(
            z=z,
            x=[f"Pred {lbl}" for lbl in labels],
            y=[f"True {lbl}" for lbl in labels],
            colorscale="Blues",
            text=z,
            texttemplate="%{text}",
            showscale=True,
        )
    )
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=20),
        height=420,
    )
    return figure


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def _render_overview(metadata: ModelMetadata) -> None:
    pills = "".join(f"<span class='pill'>{lang}</span>" for lang in metadata.supported_languages)
    authors = " · ".join(metadata.authors)
    st.markdown(
        f"""
        <div class="hero">
            <h1>{metadata.project_title}</h1>
            <p class="subtle" style="margin:0.5rem 0 0.35rem 0;">{metadata.project_description}</p>
            <p class="subtle" style="margin:0 0 0.75rem 0; font-size:0.82rem;">
                {authors} — {metadata.institution}
            </p>
            <div>{pills}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Research objectives**")
    st.markdown(
        "1. Analyse how the model interprets Filipino, English, and Taglish e-commerce "
        "reviews (code-switching, slang, sarcasm).\n"
        "2. Integrate Bayesian Weighted Sentiment scores into a recommendation framework "
        "that balances review volume and sentiment intensity.\n"
        "3. Evaluate sentiment classification into positive, negative, and neutral."
    )

    if metadata.eval_metrics:
        st.markdown("**Current held-out scores** (see **Model Results** for the full report)")
        mcols = st.columns(4)
        mcols[0].metric("Accuracy", f"{metadata.eval_metrics.get('accuracy', 0):.1%}")
        mcols[1].metric("Weighted F1", f"{metadata.eval_metrics.get('f1', 0):.3f}")
        mcols[2].metric("Precision", f"{metadata.eval_metrics.get('precision', 0):.3f}")
        mcols[3].metric("Recall", f"{metadata.eval_metrics.get('recall', 0):.3f}")

    st.divider()
    steps = [
        ("1 · Sentiment", "Classify each review as positive / negative / neutral, with a 0–1 sentiment score and confidence."),
        ("2 · Aggregation", "Group reviews by product: total reviews, average sentiment, class counts."),
        ("3 · Bayesian Score", "Weight average sentiment by review volume so high-volume products are trusted more."),
        ("4 · Ranking + RSA baseline", "Sort by Bayesian score for the Top-N, compared against a Raw Sentiment Average baseline."),
        ("5 · Recommendations", "Present a ranked table: Rank · Product · Adjusted Score · Reviews."),
        ("6 · Evaluation", "Classification metrics + ranking metrics (Precision@K, NDCG@K)."),
    ]
    cols = st.columns(3)
    for i, (title, body) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(
                f"<div class='step-card'><h4>{title}</h4><p>{body}</p></div>",
                unsafe_allow_html=True,
            )
        if i % 3 == 2:
            st.write("")

    st.divider()
    st.subheader("Methodology")
    st.markdown("**Bayesian Weighted Score (BWS)** — trusts products with more reviews (Section 3.5.2):")
    st.latex(r"BWS = \frac{\overline{s}\cdot n + m\cdot C}{n + m}")
    st.markdown("with the prior constants derived from the dataset distribution:")
    b1, b2 = st.columns(2)
    with b1:
        st.markdown("Global average sentiment (prior value):")
        st.latex(r"C = \frac{\sum_i \overline{s}_i\, n_i}{\sum_i n_i}")
    with b2:
        st.markdown("Minimum review threshold (prior weight):")
        st.latex(r"m = \text{percentile}(n,\, 25)")
    st.caption(
        "s̄ = product average sentiment score · n = product review count · "
        "C = global (review-weighted) average sentiment · m = 25th percentile of review counts. "
        "Products are sorted by BWS in descending order for the Top-N list."
    )

    st.markdown("**Ranking evaluation (Section 3.6.3)** — BWS is compared against a Raw Sentiment Average (RSA) baseline:")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Precision@K**")
        st.latex(r"\text{Precision@}K = \frac{\text{relevant items in Top-}K}{K}")
    with c2:
        st.markdown("**NDCG@K**")
        st.latex(r"NDCG@K = \frac{DCG@K}{IDCG@K}, \quad DCG@K = \sum_{i=1}^{K}\frac{2^{rel_i}-1}{\log_2(i+1)}")


def _render_single_review(service: SentimentModelService, schema_ok: bool) -> None:
    st.subheader("Single Review Sentiment")

    pool = st.session_state.get("review_pool")
    has_pool = isinstance(pool, pd.DataFrame) and not pool.empty

    source = "Type manually"
    if has_pool:
        source = st.radio(
            "Review source",
            ["Pick from dataset", "Type manually"],
            horizontal=True,
        )
    else:
        st.caption("Tip: load a reviews CSV in the **Batch & Recommendations** tab to pick real reviews here.")

    default_text = ""
    picked_rating = None
    if source == "Pick from dataset" and has_pool:
        products = ["(all products)"] + sorted(pool["product"].dropna().unique().tolist())
        f1, f2 = st.columns([3, 1])
        with f1:
            chosen_product = st.selectbox("Filter by product", products, key="single_product_filter")
        subset = pool if chosen_product == "(all products)" else pool[pool["product"] == chosen_product]
        subset = subset.reset_index(drop=True)

        with f2:
            st.write("")
            st.write("")
            if st.button("🎲 Random", width="stretch") and len(subset) > 0:
                st.session_state.picked_review_idx = int(subset.sample(1).index[0])

        if len(subset) == 0:
            st.info("No reviews for this product.")
            return

        idx = min(int(st.session_state.get("picked_review_idx", 0)), len(subset) - 1)
        options = [f"[{i}] {str(text)[:100]}" for i, text in enumerate(subset["review"].tolist())]
        selected_option = st.selectbox("Choose a review", options, index=idx, key="single_review_pick")
        idx = options.index(selected_option)
        st.session_state.picked_review_idx = idx

        default_text = str(subset["review"].iloc[idx])
        picked_rating = subset["rating"].iloc[idx] if "rating" in subset.columns else None
        st.caption(f"Product: **{subset['product'].iloc[idx]}**")
        picked_rating = subset["rating"].iloc[idx] if "rating" in subset.columns else None
    else:
        picked_rating = None

    input_text = st.text_area(
        "Review text",
        value=default_text,
        height=170,
        placeholder="Paste an English, Filipino, or Taglish (Filipino-English) review here...",
    )

    rating_value = None
    try:
        if picked_rating is not None and not pd.isna(picked_rating):
            rating_value = int(float(picked_rating))
    except (TypeError, ValueError):
        rating_value = None

    rating_value = st.number_input(
        "Star rating (optional, 1–5). Used when the checkpoint was trained with rating context.",
        min_value=0,
        max_value=5,
        value=int(rating_value) if rating_value else 0,
        help="0 = omit rating. Match the Colab format: 'Star rating: N out of 5.'",
    )
    rating_arg = rating_value if rating_value else None

    if st.button("Run Analysis", type="primary", disabled=not schema_ok):
        if not input_text or not input_text.strip():
            st.warning("Please provide non-empty text before running analysis.")
            return

        prediction = service.predict(input_text, rating=rating_arg)
        for warning in prediction.warnings:
            st.warning(warning)

        if prediction.detected_language_code == "unknown":
            st.info("Language could not be confidently detected.")
        elif not prediction.supported_language:
            st.warning(f"Detected language: {prediction.detected_language_name} (outside project sample).")
        else:
            st.success(f"Detected language: {prediction.detected_language_name}")

        left, right = st.columns([1, 1])
        with left:
            st.markdown(
                f"### Prediction: {_sentiment_badge(prediction.label)}",
                unsafe_allow_html=True,
            )
            st.metric("Confidence", f"{prediction.confidence:.4f}")
            st.progress(min(max(prediction.confidence, 0.0), 1.0))
            st.metric("Sentiment score (positive prob.)", f"{prediction.probabilities.get('positive', 0.0):.4f}")
        with right:
            st.plotly_chart(_probability_chart(prediction.probabilities), width="stretch")

        st.session_state.history.append(prediction_history_row(input_text, prediction))

    st.divider()
    st.subheader("Session History")
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), width="stretch", hide_index=True)
    else:
        st.caption("No predictions yet in this session.")


def _build_sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_title": [
                "Earbuds X", "Earbuds X", "Earbuds X",
                "Earbuds A", "Earbuds A",
                "Phone Case Y", "Phone Case Y", "Phone Case Y",
                "Power Bank Z", "Power Bank Z",
            ],
            "product_category": ["Audio"] * 5 + ["Accessories"] * 3 + ["Charging"] * 2,
            "product_review_text": [
                "Ang ganda ng sound quality, sulit na sulit!",
                "Great bass and battery life, highly recommend.",
                "Medyo mahina ang mic pero okay pa rin for the price.",
                "Okay lang, medyo mahina ang mic.",
                "Good value for money overall.",
                "Hindi maganda, madaling masira. Disappointed.",
                "Scam to, fake item, di gumana nung dumating.",
                "Sobrang nipis, scratch agad after one day.",
                "Super worth it, mabilis mag-charge, recommend ko to.",
                "Ok naman kaso matagal dumating yung parcel.",
            ],
            "Rating Star": [5, 5, 4, 3, 4, 1, 1, 2, 5, 3],
        }
    )


def _guess_column(cols, *candidates) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _index_of(options, value, default: int = 0) -> int:
    return options.index(value) if value in options else default


def _render_batch(service: SentimentModelService, schema_ok: bool) -> None:
    st.subheader("Batch Analysis & Recommendations")
    st.caption(
        "Upload your **reviews** file (review text + a product key such as `Item ID`). "
        "Optionally add a **product catalog** file to attach product names and categories "
        "via a join key. This matches your `reviews-final.csv` + `brands-final.csv` structure."
    )

    c1, c2 = st.columns(2)
    with c1:
        reviews_file = st.file_uploader("Reviews CSV (required)", type=["csv"], key="rev_csv")
    with c2:
        catalog_file = st.file_uploader("Product catalog CSV (optional)", type=["csv"], key="cat_csv")

    st.download_button(
        "Download tiny sample reviews CSV",
        data=_build_sample_dataframe().to_csv(index=False).encode("utf-8"),
        file_name="sample_reviews.csv",
        mime="text/csv",
    )
    use_sample = st.checkbox("Use built-in sample data instead", value=False)

    catalog_df: Optional[pd.DataFrame] = None
    if use_sample:
        df = _build_sample_dataframe()
    elif reviews_file is not None:
        try:
            df = pd.read_csv(reviews_file)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read reviews CSV: {exc}")
            return
        st.caption(f"Reviews: **{len(df):,}** rows · columns: {list(df.columns)}")
        if catalog_file is not None:
            try:
                catalog_df = pd.read_csv(catalog_file)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read catalog CSV: {exc}")
                return
            st.caption(f"Catalog: **{len(catalog_df):,}** rows · columns: {list(catalog_df.columns)}")
    else:
        st.info("Upload a reviews CSV (and optionally a catalog CSV), or tick the sample-data checkbox.")
        return

    if df.empty:
        st.warning("The reviews dataset is empty.")
        return

    rev_cols = list(df.columns)
    default_product_col = _guess_column(rev_cols, "product_title", "Product Name", "product", "Item ID")
    default_category_col = _guess_column(rev_cols, "product_category", "category")

    if catalog_df is not None and not catalog_df.empty:
        cat_cols = list(catalog_df.columns)
        st.markdown("**Join reviews with product catalog**")
        j1, j2 = st.columns(2)
        with j1:
            rev_key = st.selectbox(
                "Reviews join key",
                rev_cols,
                index=_index_of(rev_cols, _guess_column(rev_cols, "Item ID", "item_id", "product id")),
            )
        with j2:
            cat_key = st.selectbox(
                "Catalog join key",
                cat_cols,
                index=_index_of(cat_cols, _guess_column(cat_cols, "Item ID", "item_id", "product id")),
            )
        n1, n2 = st.columns(2)
        with n1:
            cat_name_col = st.selectbox(
                "Catalog product name column",
                cat_cols,
                index=_index_of(cat_cols, _guess_column(cat_cols, "Product Name", "product_title", "name", "title")),
            )
        with n2:
            cat_category_options = ["(none)"] + cat_cols
            cat_category_choice = st.selectbox(
                "Catalog category column (optional)",
                cat_category_options,
                index=_index_of(
                    cat_category_options,
                    _guess_column(cat_cols, "Category ID", "category", "Breadcrumbs", "Brand"),
                ),
            )

        keep = [cat_key, cat_name_col]
        if cat_category_choice != "(none)":
            keep.append(cat_category_choice)
        catalog_small = catalog_df[keep].drop_duplicates(subset=[cat_key]).copy()
        df = df.copy()
        df["_join_key"] = df[rev_key].astype(str).str.strip()
        catalog_small["_join_key"] = catalog_small[cat_key].astype(str).str.strip()
        merged = df.merge(catalog_small.drop(columns=[cat_key]), on="_join_key", how="left")

        matched = int(merged[cat_name_col].notna().sum())
        st.caption(f"Joined **{matched:,} / {len(merged):,}** reviews to a product name.")
        drop_unmatched = st.checkbox(
            "Drop reviews with no catalog match (recommended)", value=True
        )
        if drop_unmatched:
            merged = merged[merged[cat_name_col].notna()].copy()
            st.caption(f"Kept **{len(merged):,}** matched reviews for ranking.")
        else:
            merged[cat_name_col] = merged[cat_name_col].fillna("Unknown item " + merged["_join_key"])
        df = merged
        default_product_col = cat_name_col
        default_category_col = cat_category_choice if cat_category_choice != "(none)" else None

    cols = list(df.columns)
    review_guess = _guess_column(
        cols, "product_review_text", "Comment", "review", "review_text", "text", "comment"
    )
    g1, g2, g3 = st.columns(3)
    with g1:
        product_col = st.selectbox(
            "Product column (group by)", cols, index=_index_of(cols, default_product_col)
        )
    with g2:
        review_col = st.selectbox(
            "Review text column", cols, index=_index_of(cols, review_guess)
        )
    with g3:
        category_options = ["(none)"] + cols
        category_choice = st.selectbox(
            "Category column (optional)", category_options, index=_index_of(category_options, default_category_col)
        )
    category_col = None if category_choice == "(none)" else category_choice
    rating_col = "Rating Star" if "Rating Star" in df.columns else None

    if product_col in df.columns and review_col in df.columns:
        rating_col = _guess_column(cols, "Rating Star", "rating", "stars", "star")
        pool_data = {
            "product": df[product_col].astype(str),
            "review": df[review_col].astype(str),
        }
        if rating_col:
            pool_data["rating"] = df[rating_col]
        st.session_state.review_pool = pd.DataFrame(pool_data)

    n_rows = len(df)
    n_groups = df[product_col].nunique(dropna=True)
    if n_rows > 0 and n_groups / n_rows > 0.9:
        st.warning(
            f"⚠️ Grouping by **`{product_col}`** yields {n_groups:,} groups for {n_rows:,} rows "
            "— this column looks like a per-row identifier, so every product will show a review "
            "count of ~1. Pick the actual product column (e.g. **`Product Name`** after joining the "
            "catalog, or **`Item ID`**) so reviews of the same product are grouped together."
        )
    else:
        avg_reviews = n_rows / n_groups if n_groups else 0
        st.caption(
            f"Grouping by `{product_col}` → {n_groups:,} products, ~{avg_reviews:.1f} reviews each on average."
        )

    top_n = st.radio("Top-N recommendations", [5, 10], index=0, horizontal=True)

    if st.button("Run Pipeline", type="primary", disabled=not schema_ok):
        if len(df) > 5000:
            st.info(f"Processing {len(df):,} reviews — this may take a moment on CPU.")
        with st.spinner("Scoring reviews with XLM-RoBERTa..."):
            analyzed = service.analyze_reviews(df, text_col=review_col)
        aggregated = aggregate_by_product(analyzed, product_col=product_col, category_col=category_col)
        aggregated, C, m = apply_bayesian_score(aggregated)

        st.session_state.analyzed_reviews = analyzed
        st.session_state.aggregated = aggregated
        st.session_state.bayesian_params = (C, m)
        st.session_state.product_col = product_col

    aggregated = st.session_state.aggregated
    analyzed = st.session_state.analyzed_reviews
    if aggregated is None or analyzed is None:
        return

    product_col = st.session_state.product_col or product_col
    C, m = st.session_state.bayesian_params

    st.divider()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Reviews", f"{len(analyzed):,}")
    k2.metric("Products", f"{len(aggregated):,}")
    k3.metric("Global avg sentiment (C)", f"{C:.4f}")
    k4.metric("Review threshold m (p25)", f"{m:.2f}")

    dist = analyzed["sentiment_label"].value_counts().to_dict()
    d1, d2, d3 = st.columns(3)
    d1.metric("Positive", f"{dist.get('positive', 0):,}")
    d2.metric("Neutral", f"{dist.get('neutral', 0):,}")
    d3.metric("Negative", f"{dist.get('negative', 0):,}")

    st.subheader(f"Top {top_n} Recommendations")
    ranked = rank_products(aggregated, top_n=top_n)
    display = ranked.rename(
        columns={
            product_col: "Product",
            "bayesian_score": "Adjusted Score",
            "total_reviews": "Review Count",
        }
    )
    show_cols = ["rank", "Product", "Adjusted Score", "Review Count"]
    if "product_category" in display.columns:
        show_cols.insert(2, "product_category")
    st.dataframe(
        display[show_cols].rename(columns={"rank": "Rank", "product_category": "Category"}),
        width="stretch",
        hide_index=True,
    )
    st.plotly_chart(_recommendation_chart(ranked, product_col), width="stretch")

    st.subheader("BWS vs. Raw Sentiment Average (RSA) baseline")
    st.caption(
        "Paper Section 3.6.3.A — the RSA baseline ranks purely by arithmetic mean sentiment "
        "(no review-volume weighting). Comparing the two shows how BWS moderates products with "
        "very few reviews so they cannot unfairly dominate the top positions."
    )
    rsa = raw_sentiment_average_rank(aggregated, top_n=top_n)
    bws_cmp = ranked[[product_col, "avg_sentiment_score", "total_reviews", "bayesian_score"]].copy()
    bws_cmp.insert(0, "BWS Rank", range(1, len(bws_cmp) + 1))
    rsa_cmp = rsa[[product_col, "avg_sentiment_score", "total_reviews", "bayesian_score"]].copy()
    rsa_cmp.insert(0, "RSA Rank", range(1, len(rsa_cmp) + 1))

    cmp_cols = st.columns(2)
    with cmp_cols[0]:
        st.markdown("**BWS ranking (proposed)**")
        st.dataframe(
            bws_cmp.rename(
                columns={
                    product_col: "Product",
                    "avg_sentiment_score": "Raw Avg",
                    "total_reviews": "Reviews",
                    "bayesian_score": "BWS",
                }
            ),
            width="stretch",
            hide_index=True,
        )
    with cmp_cols[1]:
        st.markdown("**RSA ranking (baseline)**")
        st.dataframe(
            rsa_cmp.rename(
                columns={
                    product_col: "Product",
                    "avg_sentiment_score": "Raw Avg",
                    "total_reviews": "Reviews",
                    "bayesian_score": "BWS",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    with st.expander("Per-product aggregation (full)"):
        st.dataframe(aggregated, width="stretch", hide_index=True)
    with st.expander("Per-review sentiment (full)"):
        st.dataframe(analyzed, width="stretch", hide_index=True)

    st.download_button(
        "Download recommendations (CSV)",
        data=ranked.to_csv(index=False).encode("utf-8"),
        file_name=f"recommendations_top{top_n}.csv",
        mime="text/csv",
    )


def _render_results(service: SentimentModelService, schema_ok: bool) -> None:
    st.subheader("Held-out model results")
    payload = load_hybrid_metrics()
    if not payload:
        st.info(
            "No `hybrid_model/metrics.json` yet. Run `python train_hybrid.py` "
            "or upload a labeled CSV in the Evaluation tab."
        )
        return

    text_eval = payload.get("eval") or {}
    fused_eval = payload.get("eval_with_rating_fusion") or {}
    previous = payload.get("previous_xlmr_shopee") or {}

    st.caption(
        f"{payload.get('dataset', 'FiReCS')} · {payload.get('split', '')}. "
        f"{payload.get('baseline_note', '')}"
    )

    c1, c2, c3, c4 = st.columns(4)
    fused_acc = fused_eval.get("accuracy")
    fused_f1 = fused_eval.get("f1")
    text_acc = text_eval.get("accuracy")
    old_acc = previous.get("accuracy")
    c1.metric(
        "Accuracy (text + stars)",
        f"{fused_acc:.1%}" if fused_acc is not None else "—",
        delta=(
            f"{(fused_acc - old_acc):+.1%} vs previous XLM-R"
            if fused_acc is not None and old_acc
            else None
        ),
    )
    c2.metric(
        "Weighted F1 (text + stars)",
        f"{fused_f1:.3f}" if fused_f1 is not None else "—",
    )
    c3.metric(
        "Accuracy (text only)",
        f"{text_acc:.1%}" if text_acc is not None else "—",
    )
    c4.metric("Test reviews", f"{text_eval.get('n_samples', 0):,}")

    st.markdown("**Comparison**")
    compare = pd.DataFrame(
        [
            {
                "Model": "Previous Shopee 1k XLM-R (held-out 200)",
                "Accuracy": previous.get("accuracy"),
                "Weighted F1": previous.get("f1"),
            },
            {
                "Model": "Hybrid TF-IDF on FiReCS (text only)",
                "Accuracy": text_eval.get("accuracy"),
                "Weighted F1": text_eval.get("f1"),
            },
            {
                "Model": "Hybrid + Shopee star-rating fusion",
                "Accuracy": fused_eval.get("accuracy"),
                "Weighted F1": fused_eval.get("f1"),
            },
        ]
    )
    st.dataframe(compare, width="stretch", hide_index=True)

    chart = go.Figure()
    names = compare["Model"].tolist()
    accs = [v if v is not None else 0 for v in compare["Accuracy"].tolist()]
    f1s = [v if v is not None else 0 for v in compare["Weighted F1"].tolist()]
    chart.add_trace(go.Bar(name="Accuracy", x=names, y=accs, marker_color="#2ECC71"))
    chart.add_trace(go.Bar(name="Weighted F1", x=names, y=f1s, marker_color="#5DADE2"))
    chart.update_layout(
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, 1], title="Score"),
        height=360,
        margin=dict(l=20, r=20, t=30, b=80),
    )
    st.plotly_chart(chart, width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix (text + stars)**")
        cm = fused_eval.get("confusion_matrix") or text_eval.get("confusion_matrix")
        labels = fused_eval.get("labels") or text_eval.get("labels") or LABEL_ORDER
        if cm is not None:
            st.plotly_chart(_confusion_matrix_chart(np.array(cm), labels), width="stretch")
    with right:
        st.markdown("**Per-class F1 (text + stars)**")
        rows = fused_eval.get("per_class") or text_eval.get("per_class") or []
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.divider()
    st.markdown("### Live Taglish predictions")
    st.caption("The same hybrid model running inside this app, with star-rating fusion.")
    demos = [
        ("Ang ganda ng sound quality, sulit na sulit!", 5, "positive"),
        ("Okay lang, medyo mahina ang mic.", 3, "neutral"),
        ("Hindi maganda, madaling masira. Disappointed.", 1, "negative"),
        ("Ganda sana kaso delayed yung delivery and may scratch.", 3, "neutral"),
        ("Super worth it, mabilis mag-charge, recommend ko to.", 5, "positive"),
        ("Scam to, fake item, di gumana nung dumating.", 1, "negative"),
    ]
    if not schema_ok:
        st.warning("Model is not loaded, so live predictions are disabled.")
        return

    records = []
    for text, stars, expected in demos:
        prediction = service.predict(text, rating=stars)
        records.append(
            {
                "Review": text,
                "Stars": stars,
                "Predicted": prediction.label,
                "Confidence": round(prediction.confidence, 3),
                "P(neg)": round(prediction.probabilities.get("negative", 0), 3),
                "P(neu)": round(prediction.probabilities.get("neutral", 0), 3),
                "P(pos)": round(prediction.probabilities.get("positive", 0), 3),
            }
        )
    st.dataframe(pd.DataFrame(records), width="stretch", hide_index=True)


def _render_evaluation(service: SentimentModelService, schema_ok: bool) -> None:
    st.subheader("Model Evaluation")

    st.markdown("### Evaluation 1 — Classification Metrics")
    st.caption(
        "Upload a labeled CSV with a text column and a true-label column "
        "(labels as 0/1/2 or negative/neutral/positive)."
    )
    labeled_file = st.file_uploader("Upload labeled test CSV", type=["csv"], key="eval_csv")
    if labeled_file is not None:
        try:
            eval_df = pd.read_csv(labeled_file)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read CSV: {exc}")
            eval_df = None

        if eval_df is not None and not eval_df.empty:
            cols = list(eval_df.columns)
            e1, e2 = st.columns(2)
            with e1:
                text_col = st.selectbox("Text column", cols, key="eval_text_col")
            with e2:
                label_col = st.selectbox("True label column", cols, key="eval_label_col")

            if st.button("Run Classification Evaluation", type="primary", disabled=not schema_ok):
                with st.spinner("Evaluating on labeled data..."):
                    try:
                        rating_col = _guess_column(
                            cols, "Rating Star", "rating", "stars", "star"
                        )
                        ratings = (
                            eval_df[rating_col].tolist() if rating_col else None
                        )
                        result = service.evaluate_classification(
                            eval_df[text_col].fillna("").astype(str).tolist(),
                            eval_df[label_col].tolist(),
                            ratings=ratings,
                        )
                    except ValueError as exc:
                        st.error(str(exc))
                        result = None

                if result is not None:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Accuracy", f"{result['accuracy']:.4f}")
                    m2.metric("Precision (w)", f"{result['precision']:.4f}")
                    m3.metric("Recall (w)", f"{result['recall']:.4f}")
                    m4.metric("F1 (w)", f"{result['f1']:.4f}")

                    cc1, cc2 = st.columns([1, 1])
                    with cc1:
                        st.plotly_chart(
                            _confusion_matrix_chart(result["confusion_matrix"], result["labels"]),
                            width="stretch",
                        )
                    with cc2:
                        st.markdown("**Per-class report**")
                        st.dataframe(
                            pd.DataFrame(result["per_class"]),
                            width="stretch",
                            hide_index=True,
                        )
                    st.caption(f"Evaluated on {result['n_samples']:,} samples.")

    st.divider()
    st.markdown("### Evaluation 2 — Ranking Metrics (Precision@K, NDCG@K)")
    aggregated = st.session_state.aggregated
    product_col = st.session_state.product_col
    if aggregated is None or product_col is None:
        st.info("Run the pipeline in the **Batch & Recommendations** tab first.")
        return

    st.caption(
        "Per the paper, a product is relevant when it has a high adjusted sentiment score "
        "and a sufficient review count. Set those thresholds below. The proposed BWS ranking "
        "is compared against the Raw Sentiment Average (RSA) baseline."
    )
    r1, r2, r3 = st.columns(3)
    with r1:
        sentiment_threshold = st.slider("Relevance: min avg sentiment", 0.0, 1.0, 0.6, 0.05)
    with r2:
        min_reviews = st.number_input("Relevance: min reviews", min_value=1, value=5, step=1)
    with r3:
        ks_text = st.text_input("K values (comma-separated)", value="5, 10, 20")

    try:
        ks = [int(x.strip()) for x in ks_text.split(",") if x.strip()]
    except ValueError:
        st.error("K values must be integers.")
        ks = [5, 10, 20]

    ranking = evaluate_ranking(
        aggregated,
        product_col=product_col,
        ks=ks,
        sentiment_threshold=sentiment_threshold,
        min_reviews=int(min_reviews),
        score_col="bayesian_score",
    )
    ranking_rsa = evaluate_ranking(
        aggregated,
        product_col=product_col,
        ks=ks,
        sentiment_threshold=sentiment_threshold,
        min_reviews=int(min_reviews),
        score_col="avg_sentiment_score",
    )

    e1, e2 = st.columns(2)
    e1.metric("Total products", f"{ranking['total_products']:,}")
    e2.metric("Relevant products", f"{ranking['relevant_products']:,}")

    if ranking["results"]:
        bws_df = pd.DataFrame(ranking["results"])
        rsa_df = pd.DataFrame(ranking_rsa["results"])
        merged = bws_df.merge(rsa_df, on="K", suffixes=(" (BWS)", " (RSA)"))
        merged = merged[
            ["K", "Precision@K (BWS)", "Precision@K (RSA)", "NDCG@K (BWS)", "NDCG@K (RSA)"]
        ]
        st.dataframe(merged, width="stretch", hide_index=True)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=bws_df["K"], y=bws_df["Precision@K"], mode="lines+markers", name="Precision@K (BWS)")
        )
        fig.add_trace(
            go.Scatter(x=rsa_df["K"], y=rsa_df["Precision@K"], mode="lines+markers",
                       name="Precision@K (RSA)", line=dict(dash="dash"))
        )
        fig.add_trace(
            go.Scatter(x=bws_df["K"], y=bws_df["NDCG@K"], mode="lines+markers", name="NDCG@K (BWS)")
        )
        fig.add_trace(
            go.Scatter(x=rsa_df["K"], y=rsa_df["NDCG@K"], mode="lines+markers",
                       name="NDCG@K (RSA)", line=dict(dash="dash"))
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="K",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1.05]),
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("Not enough products to compute ranking metrics.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    _inject_theme()
    _init_session_state()

    with st.sidebar:
        st.title("Controls")
        notebook_path = st.text_input(
            "Notebook path",
            value="pipeline.ipynb",
            help="Project notebook used to extract model/training metadata.",
        )
        model_dir = st.text_input(
            "Fine-tuned model directory (optional)",
            value="",
            help="Leave blank to auto-resolve models/xlmr_sentiment_model.",
        )

    try:
        metadata, service, model_source, schema_ok, schema_message = load_service(
            notebook_path=notebook_path, model_dir=model_dir
        )
    except Exception as exc:  # noqa: BLE001
        st.error("Failed to load the model.")
        st.exception(exc)
        st.stop()

    _render_sidebar(metadata, service, model_source, schema_ok)

    if not schema_ok:
        st.error(
            "⚠️ No usable sentiment model is loaded, so predictions are disabled. "
            "Keep `hybrid_model/` in the repo (or train it with `python train_hybrid.py`), "
            "or place a fine-tuned XLM-RoBERTa checkpoint at `models/xlmr_sentiment_model`."
        )
        st.caption(schema_message)

    tab_overview, tab_results, tab_single, tab_batch, tab_eval = st.tabs(
        ["Overview", "Model Results", "Single Review", "Batch & Recommendations", "Evaluation"]
    )
    with tab_overview:
        _render_overview(metadata)
    with tab_results:
        _render_results(service, schema_ok)
    with tab_batch:
        _render_batch(service, schema_ok)
    with tab_single:
        _render_single_review(service, schema_ok)
    with tab_eval:
        _render_evaluation(service, schema_ok)

    st.markdown(
        f"<div class='footer'>{metadata.project_title} · "
        f"{' · '.join(metadata.authors)} · {metadata.institution} · "
        "XLM-RoBERTa · Bayesian Weighted Ranking</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
