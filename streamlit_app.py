"""
Anomalous ≠ Malicious — an interactive look at why unsupervised intrusion
detection plateaus on real botnet traffic.

Run locally:   streamlit run streamlit_app.py
Deploy:        push to GitHub → share.streamlit.io → point at this file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import ks_2samp
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Anomalous ≠ Malicious", page_icon="🛰️", layout="wide")

SAMPLE_PATH = "sample_logs/ctu13_sample.csv"
DROP_COLS = {"Unnamed: 0", "Label", "true_label"}

C_BENIGN = "#2196F3"
C_ATTACK = "#E53935"


# ── data ──────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PATH)


def make_synthetic(n: int = 4000) -> pd.DataFrame:
    """Scores-only synthetic set that mimics the real overlap (for the no-data demo)."""
    rng = np.random.default_rng(42)
    benign = rng.normal(0.0, 1.0, n)
    obvious = rng.normal(-3.0, 0.7, n // 4)
    stealthy = rng.normal(-0.3, 1.0, n - n // 4)  # hides inside the benign pile
    df = pd.DataFrame({
        "f1": np.concatenate([benign, obvious, stealthy]),
        "f2": rng.normal(0, 1, n * 2),
        "true_label": np.concatenate([np.zeros(n), np.ones(n)]).astype(int),
    })
    return df.sample(frac=1, random_state=1).reset_index(drop=True)


def featurise(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y = df["true_label"].astype(int).values
    feat_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feat_cols].select_dtypes(include=[np.number]).copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(numeric_only=True), inplace=True)
    X = X.loc[:, X.var() > 0]
    skewed = X.skew()[lambda s: s > 2].index
    if len(skewed):
        X[skewed] = np.log1p(X[skewed].clip(lower=0))
    return StandardScaler().fit_transform(X.values), y


@st.cache_data(show_spinner=False)
def run_if(X: np.ndarray, y: np.ndarray, contamination: float):
    model = IsolationForest(n_estimators=200, contamination=contamination,
                            random_state=42, n_jobs=-1)
    model.fit(X)
    scores = model.score_samples(X)          # lower = more anomalous
    pred = (model.predict(X) == -1).astype(int)
    auc = roc_auc_score(y, -scores)
    ks = ks_2samp(scores[y == 1], scores[y == 0]).statistic
    cm = confusion_matrix(y, pred)
    return scores, pred, auc, ks, cm


# ── charts ────────────────────────────────────────────────────────────────────

def piles_fig(scores, y):
    fig = go.Figure()
    fig.add_histogram(x=scores[y == 0], name="Benign", histnorm="probability density",
                      opacity=0.6, marker_color=C_BENIGN, nbinsx=60)
    fig.add_histogram(x=scores[y == 1], name="Malicious", histnorm="probability density",
                      opacity=0.6, marker_color=C_ATTACK, nbinsx=60)
    fig.update_layout(barmode="overlay", height=380,
                      margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.1),
                      xaxis_title="Isolation Forest anomaly score  (lower = more anomalous)",
                      yaxis_title="density")
    return fig


def cdf_fig(scores, y, ks):
    grid = np.linspace(np.percentile(scores, 0.5), np.percentile(scores, 99.5), 400)
    cdf_b = np.searchsorted(np.sort(scores[y == 0]), grid, side="right") / (y == 0).sum()
    cdf_a = np.searchsorted(np.sort(scores[y == 1]), grid, side="right") / (y == 1).sum()
    k = int(np.argmax(np.abs(cdf_a - cdf_b)))
    fig = go.Figure()
    fig.add_scatter(x=grid, y=cdf_b, name="Benign CDF", line=dict(color=C_BENIGN, width=3))
    fig.add_scatter(x=grid, y=cdf_a, name="Malicious CDF", line=dict(color=C_ATTACK, width=3))
    fig.add_scatter(x=[grid[k], grid[k]], y=[cdf_b[k], cdf_a[k]], mode="lines+markers",
                    name=f"KS = {ks:.3f}", line=dict(color="#FB8C00", width=3, dash="dot"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.1),
                      xaxis_title="Isolation Forest anomaly score",
                      yaxis_title="cumulative fraction")
    return fig


# ── app ───────────────────────────────────────────────────────────────────────

st.title("🛰️ Anomalous ≠ Malicious")
st.caption(
    "Why unsupervised intrusion detection plateaus on real botnet traffic — "
    "an interactive diagnosis on the CTU-13 dataset."
)

with st.sidebar:
    st.header("Controls")
    source = st.radio(
        "Data",
        ["CTU-13 sample (real botnet traffic)", "Upload a CSV", "Synthetic demo"],
    )
    contamination = st.slider(
        "Contamination (threshold)", 0.05, 0.50, 0.45, 0.01,
        help="Fraction of flows the detector flags. Moves the threshold, NOT the score ranking — "
             "watch recall/precision change while AUC and KS stay put.",
    )
    st.markdown("---")
    st.markdown(
        "**The idea:** Isolation Forest ranks flows by *statistical rarity*. "
        "But stealthy C&C traffic isn't rare, and big benign transfers are — so the "
        "two score distributions **overlap**, and no threshold cleanly separates them."
    )

# Resolve the dataframe
if source.startswith("Upload"):
    up = st.file_uploader("CSV with a `true_label` or `Label` column (1 = attack, 0 = benign)", type="csv")
    if up is None:
        st.info("Upload a CTU-13-format CSV to begin, or pick another data source.")
        st.stop()
    df = pd.read_csv(up)
    if "true_label" not in df.columns:
        if "Label" in df.columns and set(df["Label"].unique()) <= {0, 1}:
            df["true_label"] = df["Label"]
        else:
            st.error("Couldn't find a binary `true_label`/`Label` column (values 1/0).")
            st.stop()
elif source.startswith("Synthetic"):
    df = make_synthetic()
else:
    df = load_sample()

X, y = featurise(df)
scores, pred, auc, ks, cm = run_if(X, y, contamination)

# Metrics
tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp) if (tp + fp) else 0.0
recall = tp / (tp + fn) if (tp + fn) else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("AUC", f"{auc:.3f}", help="P(attack scores more anomalous than benign). 0.5 = no signal.")
c2.metric("KS distance", f"{ks:.3f}", help="Separation between the two score piles = the ceiling for any threshold.")
c3.metric("Attack recall", f"{recall:.0%}", help="Share of real attacks caught at this threshold.")
c4.metric("Attack precision", f"{precision:.0%}")

st.markdown("---")
left, right = st.columns(2)
with left:
    st.subheader("The two score piles")
    st.plotly_chart(piles_fig(scores, y), width='stretch')
    st.caption("Where blue and red overlap, benign and malicious flows get the **same** score — unseparable.")
with right:
    st.subheader("The ceiling (KS)")
    st.plotly_chart(cdf_fig(scores, y, ks), width='stretch')
    st.caption("The orange gap (**KS**) is the best (recall − false-alarm) **any** threshold can achieve.")

with st.expander("Confusion matrix at this threshold"):
    cm_df = pd.DataFrame(cm, index=["Actual benign", "Actual attack"],
                         columns=["Pred benign", "Pred attack"])
    st.dataframe(cm_df, width='stretch')

st.markdown("---")
st.markdown(
    f"""
**Read it:** at contamination **{contamination:.0%}**, this detector catches **{recall:.0%}** of
real botnet attacks at **{precision:.0%}** precision, with **zero labels used in training**.
Slide the threshold and watch recall/precision trade off — but **AUC ({auc:.3f}) and KS ({ks:.3f})
barely move**, because they depend on the *score ranking*, not the threshold. That's the proof the
ceiling is **structural**: the fix isn't a better threshold, it's a better representation
(timing / graph features) that pulls the two piles apart.

Built on [github.com/akrishnash/anomaly-detection](https://github.com/akrishnash/anomaly-detection)
· detector: scikit-learn Isolation Forest · agent layer: GPT-4o tool calling.
"""
)
