import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import uuid

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Fraud Risk Command Center", page_icon="🛡️", layout="wide")

# Design system 
st.markdown("""
<style>
.stApp { background-color: #0e1117; }
.kpi-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 12px;
    padding: 1.2rem 1.4rem; margin-bottom: 0.5rem;
}
.kpi-value { font-size: 2rem; font-weight: 700; color: #f0f6fc; }
.kpi-label { font-size: 0.85rem; color: #8b949e; margin-bottom: 0.3rem; }
.kpi-sub { font-size: 0.78rem; color: #58a6ff; margin-top: 0.2rem; }
.risk-card-high { background: #2d1214; border-left: 5px solid #f85149; border-radius: 10px; padding: 1.5rem; }
.risk-card-med { background: #2d2410; border-left: 5px solid #d29922; border-radius: 10px; padding: 1.5rem; }
.risk-card-low { background: #12261a; border-left: 5px solid #3fb950; border-radius: 10px; padding: 1.5rem; }
.signal-badge {
    display: inline-block; background: #21262d; border: 1px solid #30363d;
    padding: 4px 12px; border-radius: 14px; margin: 3px; font-size: 0.82rem; color: #c9d1d9;
}
.status-dot { color: #3fb950; font-size: 0.85rem; }
.econ-strip { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 1rem 1.4rem; }
</style>
""", unsafe_allow_html=True)

# ---------- Session state ----------
if "cases" not in st.session_state:
    st.session_state.cases = [
        {"transaction_id": "DEMO-1042", "risk_score": 0.94, "amount": 18500, "reason": "Unusual amount + high velocity", "status": "Pending", "timestamp": "Demo case", "is_demo": True},
        {"transaction_id": "DEMO-1043", "risk_score": 0.87, "amount": 7200, "reason": "High-risk category", "status": "Confirmed Fraud", "timestamp": "Demo case", "is_demo": True},
        {"transaction_id": "DEMO-1044", "risk_score": 0.21, "amount": 1200, "reason": "Normal behavior", "status": "Approved", "timestamp": "Demo case", "is_demo": True},
    ]
if "last_result" not in st.session_state:
    st.session_state.last_result = None


def readable_feature(f: dict) -> str:
    name = f["feature"]
    if name.startswith("category_"):
        return f"Category: {name.replace('category_', '').replace('_', ' ').title()}"
    if name == "amt":
        return "Unusual Amount"
    if name == "amt_zscore_per_card":
        return "Spend Anomaly"
    if name == "hour":
        return "Time of Day"
    if name == "seconds_since_last_trans":
        return "Transaction Velocity"
    return name.replace("_", " ").title()


def risk_bucket(prob: float) -> tuple[str, str]:
    if prob >= 0.7:
        return "HIGH RISK", "risk-card-high"
    if prob >= 0.35:
        return "MEDIUM RISK", "risk-card-med"
    return "LOW RISK", "risk-card-low"


tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Transaction Review", "Case Queue", "Model Performance"])

# ============================================================
# TAB 1 — OVERVIEW
# ============================================================
with tab1:
    st.markdown("### Fraud Risk Command Center")
    st.caption("AI-powered transaction monitoring & fraud prevention")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3)
        if health.status_code == 200:
            st.markdown('<span class="status-dot">●</span> SYSTEM OPERATIONAL', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#f85149;">●</span> RISK ENGINE DEGRADED', unsafe_allow_html=True)
    except requests.exceptions.RequestException:
        st.markdown('<span style="color:#f85149;">●</span> RISK ENGINE OFFLINE', unsafe_allow_html=True)
    st.write("")

    PRECISION, RECALL, PR_AUC, THRESHOLD = 0.30, 0.965, 0.909, 0.35
    F1 = round(2 * PRECISION * RECALL / (PRECISION + RECALL), 3)

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, sub in [
        (c1, "Total Transactions Scored", "555,719", "Held-out test set"),
        (c2, "Fraud Caught", "~2,070", "of 2,145 fraud cases · 96.5% recall"),
        (c3, "Current Threshold", f"{THRESHOLD:.2f}", "Decision threshold"),
        (c4, "PR-AUC", f"{PR_AUC:.3f}", "Threshold-independent"),
    ]:
        with col:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class="econ-strip">
        <b>Break-even precision:</b> 11.25% &nbsp;|&nbsp;
        <b>Operating point:</b> 2.6× above break-even<br>
        <span style="color:#8b949e; font-size:0.85rem;">
        The model currently operates well above the precision required to break even on fraud losses.
        </span>
    </div>""", unsafe_allow_html=True)
    st.write("")

    colA, colB = st.columns([3, 2])

    # * fetch once, reuse for both charts — was hitting the API twice per render
    segments_resp, segments_error = None, None
    try:
        r = requests.get(f"{API_URL}/risk-segments", timeout=5)
        segments_resp = r.json()["elevated_risk_categories"]
    except requests.exceptions.RequestException:
        segments_error = "Risk engine unavailable. Please verify that the FastAPI service is running."

    with colA:
        st.markdown("**Fraud Risk Trend**")
        st.caption("Simulated monitoring view based on held-out test data")
        if segments_error:
            st.error(segments_error)
        else:
            df = pd.DataFrame(segments_resp)
            dates = pd.date_range(end=datetime.now(), periods=14)
            trend = pd.DataFrame({
                "date": dates,
                "fraud_rate": (df["baseline_mean"].mean() * (1 + 0.3 * pd.Series(range(14)).apply(lambda x: (-1) ** x * 0.1))).values
            })
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=trend["date"], y=trend["fraud_rate"] * 100,
                                      mode="lines", fill="tozeroy", line=dict(color="#f85149")))
            fig.update_layout(template="plotly_dark", plot_bgcolor="#161b22", paper_bgcolor="#161b22",
                               height=320, margin=dict(l=10, r=10, t=10, b=10),
                               xaxis_title=None, yaxis_title="Fraud rate %")
            st.plotly_chart(fig, use_container_width=True)

    with colB:
        st.markdown("**Elevated Risk by Category**")
        if segments_error:
            st.error(segments_error)
        else:
            df = pd.DataFrame(segments_resp).sort_values("baseline_mean", ascending=True)
            fig2 = px.bar(df, x="baseline_mean", y="category", orientation="h",
                          color="baseline_mean", color_continuous_scale="Reds")
            fig2.update_layout(template="plotly_dark", plot_bgcolor="#161b22", paper_bgcolor="#161b22",
                                height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                                coloraxis_showscale=False, xaxis_title="Baseline fraud rate", yaxis_title=None)
            st.plotly_chart(fig2, use_container_width=True)
            st.caption(f"{len(df)} categories monitored for elevated fraud risk")

    st.write("")
    donut_col, gauge_col = st.columns(2)

    with donut_col:
        st.markdown("**Fraud vs. Legitimate Transactions**")
        st.caption("Test set: 555,719 transactions, 2,145 confirmed fraud")
        FRAUD_COUNT, LEGIT_COUNT = 2145, 553574
        donut_fig = go.Figure(data=[go.Pie(
            labels=["Legitimate", "Fraud"],
            values=[LEGIT_COUNT, FRAUD_COUNT],
            hole=0.65,
            marker=dict(colors=["#3fb950", "#f85149"]),
            textinfo="label+percent",
            textfont=dict(color="white"),
        )])
        donut_fig.update_layout(
            template="plotly_dark", plot_bgcolor="#161b22", paper_bgcolor="#161b22",
            height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
            annotations=[dict(text="0.39%<br>fraud", x=0.5, y=0.5, font_size=18, font_color="#f0f6fc", showarrow=False)],
        )
        st.plotly_chart(donut_fig, use_container_width=True)

    with gauge_col:
        st.markdown("**Recall at Current Threshold**")
        st.caption(f"Fraud caught vs. missed, at threshold {THRESHOLD:.2f}")
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=RECALL * 100,
            number={"suffix": "%", "font": {"color": "#f0f6fc"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
                "bar": {"color": "#58a6ff"},
                "bgcolor": "#161b22",
                "steps": [
                    {"range": [0, 50], "color": "#2d1214"},
                    {"range": [50, 80], "color": "#2d2410"},
                    {"range": [80, 100], "color": "#12261a"},
                ],
            },
        ))
        gauge_fig.update_layout(template="plotly_dark", paper_bgcolor="#161b22",
                                 height=300, margin=dict(l=20, r=20, t=20, b=10))
        st.plotly_chart(gauge_fig, use_container_width=True)

    st.write("")
    st.markdown("**AI Model Health**")
    h1, h2, h3, h4 = st.columns(4)
    for col, label, val in [(h1, "Precision", PRECISION), (h2, "Recall", RECALL),
                             (h3, "F1", F1), (h4, "PR-AUC", PR_AUC)]:
        with col:
            st.progress(min(val, 1.0), text=f"{label}: {val:.1%}" if val <= 1 else f"{label}: {val}")
    st.caption(f"Current operating threshold: {THRESHOLD:.2f}")

# TAB 2 — TRANSACTION REVIEW
with tab2:
    st.markdown("### Transaction Review")
    st.caption("Score a transaction and understand why the model flagged it.")

    with st.form("score_form"):
        s1, s2, s3 = st.columns(3)
        with s1:
            amt = st.number_input("Amount (₹)", min_value=0.0, value=850.0)
            category = st.selectbox("Category", [
                "shopping_net", "shopping_pos", "grocery_pos", "grocery_net",
                "gas_transport", "misc_net", "misc_pos", "entertainment",
                "food_dining", "health_fitness", "home", "kids_pets",
                "personal_care", "travel"
            ])
            gender = st.selectbox("Gender", ["M", "F"])
        with s2:
            trans_time = st.text_input("Transaction time (ISO)", value=datetime.now().isoformat())
            dob = st.text_input("Date of birth (ISO)", value="1988-03-09T00:00:00")
            city_pop = st.number_input("City population", value=3495)
        with s3:
            card_amt_mean = st.number_input("Card's historical avg spend", value=120.0)
            card_amt_std = st.number_input("Card's historical spend std dev", value=45.0)

        with st.expander("Location details (advanced)"):
            l1, l2 = st.columns(2)
            with l1:
                lat = st.number_input("Cardholder lat", value=36.07)
                long = st.number_input("Cardholder long", value=-81.17)
            with l2:
                merch_lat = st.number_input("Merchant lat", value=36.01)
                merch_long = st.number_input("Merchant long", value=-82.04)

        submitted = st.form_submit_button("Run Risk Assessment", type="primary", use_container_width=True)

    if submitted:
        payload = {
            "trans_date_trans_time": trans_time, "cc_num": "demo", "category": category,
            "amt": amt, "gender": gender, "dob": dob, "lat": lat, "long": long,
            "merch_lat": merch_lat, "merch_long": merch_long, "city_pop": city_pop,
            "card_amt_mean": card_amt_mean, "card_amt_std": card_amt_std,
        }
        try:
            with st.spinner("Running risk assessment..."):
                resp = requests.post(f"{API_URL}/score", json=payload, timeout=15)
            if resp.status_code == 200:
                st.session_state.last_result = {**resp.json(), "amount": amt, "category": category}
            else:
                st.error(f"Risk engine returned an error ({resp.status_code}). Please check the transaction details.")
        except requests.exceptions.RequestException:
            st.error("Risk engine unavailable. Please verify that the FastAPI service is running.")

    result = st.session_state.last_result
    if result:
        prob = result["fraud_probability"]
        label, css_class = risk_bucket(prob)

        st.write("")
        st.markdown(f"""<div class="{css_class}">
            <div style="font-size:0.9rem; letter-spacing:1px; color:#8b949e;">{label}</div>
            <div style="font-size:2.2rem; font-weight:700;">{prob:.1%} Fraud Probability</div>
            <div style="font-size:1rem; margin-top:0.3rem;">
                {"🚩 FLAGGED" if result["is_flagged"] else "✅ NOT FLAGGED"}
                <span style="color:#8b949e;"> · threshold {result['threshold_used']:.0%}</span>
            </div>
        </div>""", unsafe_allow_html=True)

        st.write("")
        st.markdown("**Why was this transaction flagged?**")
        if result.get("risk_signals"):
            for s in result["risk_signals"]:
                st.markdown(f"- {s}")
        elif not result["is_flagged"]:
            st.caption("Transaction was not flagged — no risk signals to show.")

        if result.get("plain_explanation"):
            source = result.get("explanation_source")
            exp_label = "🤖 AI-Generated Explanation" if source == "ai" else "📐 Model-Based Explanation (AI unavailable)"
            st.info(f"**{exp_label}**\n\n{result['plain_explanation']}")

        if result.get("top_features"):
            st.write("**Contributing model features**")
            st.caption("Global feature importance — not unique to this transaction")
            badges = "".join(f'<span class="signal-badge">{readable_feature(f)}</span>' for f in result["top_features"])
            st.markdown(badges, unsafe_allow_html=True)

        st.write("")
        expected_loss = result["amount"] * prob
        protected_value = result["amount"] * prob if result["is_flagged"] else 0.0
        i1, i2 = st.columns(2)
        with i1:
            st.metric("Expected loss if unblocked", f"₹{expected_loss:,.0f}",
                       help="Transaction amount × model's fraud probability")
        with i2:
            st.metric("Potential value protected", f"₹{protected_value:,.0f}",
                       help="Only counted when the transaction is flagged and blocked")

        st.write("")
        d1, d2, _ = st.columns([1, 1, 3])
        with d1:
            if st.button("✅ Approve", use_container_width=True):
                st.session_state.cases.append({
                    "transaction_id": f"TXN-{str(uuid.uuid4())[:6].upper()}",
                    "risk_score": prob, "amount": result["amount"],
                    "reason": readable_feature(result["top_features"][0]) if result.get("top_features") else "N/A",
                    "status": "Approved", "timestamp": datetime.now().strftime("%H:%M:%S"), "is_demo": False,
                })
                st.success("Case logged as Approved.")
        with d2:
            if st.button("🚫 Confirm Fraud", use_container_width=True):
                st.session_state.cases.append({
                    "transaction_id": f"TXN-{str(uuid.uuid4())[:6].upper()}",
                    "risk_score": prob, "amount": result["amount"],
                    "reason": readable_feature(result["top_features"][0]) if result.get("top_features") else "N/A",
                    "status": "Confirmed Fraud", "timestamp": datetime.now().strftime("%H:%M:%S"), "is_demo": False,
                })
                st.success("Case logged as Confirmed Fraud.")

# TAB 3 — CASE QUEUE
with tab3:
    st.markdown("### Case Queue")
    st.caption("Review and manage transactions assessed during this session.")

    cases = st.session_state.cases
    total = len(cases)
    pending = sum(1 for c in cases if c["status"] == "Pending")
    approved = sum(1 for c in cases if c["status"] == "Approved")
    fraud = sum(1 for c in cases if c["status"] == "Confirmed Fraud")

    q1, q2, q3, q4 = st.columns(4)
    for col, label, val in [(q1, "Total Cases", total), (q2, "Pending", pending),
                             (q3, "Approved", approved), (q4, "Confirmed Fraud", fraud)]:
        with col:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
            </div>""", unsafe_allow_html=True)

        st.write("")
    if cases:
        df = pd.DataFrame(cases)
        df["risk_score"] = (df["risk_score"] * 100).round(1)
        df["amount"] = df["amount"].astype(float).round(2)
        df["Source"] = df["is_demo"].map({True: "🧪 Demo", False: "✅ Live"})
        df = df.drop(columns=["is_demo"]).rename(columns={
            "transaction_id": "Transaction", "risk_score": "Risk %", "amount": "Amount (₹)",
            "reason": "Reason", "status": "Status", "timestamp": "Time"
        }).sort_values("Risk %", ascending=False)
        st.dataframe(
            df, use_container_width=True, hide_index=True,
            column_config={
                "Amount (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                "Risk %": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )
        live_count = sum(1 for c in cases if not c["is_demo"])
        st.caption(f"{live_count} live case(s) scored this session, {len(cases) - live_count} demo case(s) for illustration.")
    else:
        st.caption("No cases yet — score a transaction in Transaction Review.")

# TAB 4 — MODEL PERFORMANCE

with tab4:
    st.markdown("### Model Performance")
    st.caption("Held-out test set (555,719 transactions, 2,145 confirmed fraud) — XGBoost, from the training notebook")

    # --- Confusion matrix at the production threshold (0.35) ---
    # Derived directly from the notebook's classification_report at threshold≈0.3488:
    # precision=0.300, recall=0.965, support=2145 fraud / 553574 legit
    TP, FN, FP, TN = 2070, 75, 4830, 548744

    st.markdown("**Confusion Matrix** — threshold 0.35")
    cm_col, sig_col = st.columns([1, 1])

    with cm_col:
        cm_fig = go.Figure(data=go.Heatmap(
            z=[[TN, FP], [FN, TP]],
            x=["Predicted Legit", "Predicted Fraud"],
            y=["Actual Legit", "Actual Fraud"],
            text=[[f"{TN:,}", f"{FP:,}"], [f"{FN:,}", f"{TP:,}"]],
            texttemplate="%{text}",
            textfont={"size": 16, "color": "white"},
            colorscale=[[0, "#161b22"], [1, "#3fb950"]],
            showscale=False,
        ))
        cm_fig.update_layout(template="plotly_dark", plot_bgcolor="#161b22", paper_bgcolor="#161b22",
                              height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(cm_fig, use_container_width=True)
        st.caption(f"Catches {TP:,}/{TP+FN:,} fraud cases (96.5% recall) at 30.0% precision")

    with sig_col:
        st.markdown("**What separates fraud from legit transactions**")
        st.caption("Mean feature values, fraud vs. legitimate (train set)")
        signal_fig = go.Figure()
        signal_fig.add_trace(go.Bar(
            y=["Spend anomaly (z-score)", "Time since last txn (sec)", "Distance to merchant (km)"],
            x=[3.204, 21276, 76.27], name="Fraud", orientation="h", marker_color="#f85149"
        ))
        signal_fig.add_trace(go.Bar(
            y=["Spend anomaly (z-score)", "Time since last txn (sec)", "Distance to merchant (km)"],
            x=[-0.019, 32538, 76.11], name="Legit", orientation="h", marker_color="#3fb950"
        ))
        signal_fig.update_layout(template="plotly_dark", plot_bgcolor="#161b22", paper_bgcolor="#161b22",
                                  height=320, margin=dict(l=10, r=10, t=10, b=10), barmode="group",
                                  xaxis_title=None)
        st.plotly_chart(signal_fig, use_container_width=True)
        st.caption("Spend anomaly and transaction velocity separate fraud clearly; merchant distance barely differs — it isn't a strong signal for this dataset.")

    st.write("")
    st.markdown("**Precision–Recall Trade-off**")
    st.caption("Sampled operating points from the test-set PR curve (not the full curve — only these 5 thresholds were evaluated)")

    pr_points = pd.DataFrame({
        "threshold": [0.9822, 0.8880, 0.6947, 0.3488, 0.0512],
        "precision": [0.900, 0.700, 0.500, 0.300, 0.113],
        "recall": [0.795, 0.890, 0.931, 0.965, 0.990],
    })
    pr_fig = go.Figure()
    pr_fig.add_trace(go.Scatter(x=pr_points["recall"], y=pr_points["precision"],
                                 mode="lines+markers", line=dict(color="#58a6ff"),
                                 marker=dict(size=10),
                                 text=[f"t={t:.2f}" for t in pr_points["threshold"]],
                                 hovertemplate="Recall %{x:.1%}<br>Precision %{y:.1%}<br>%{text}<extra></extra>"))
    pr_fig.add_vline(x=0.965, line_dash="dash", line_color="#f85149",
                      annotation_text="Current operating point (t=0.35)", annotation_position="top left")
    pr_fig.update_layout(template="plotly_dark", plot_bgcolor="#161b22", paper_bgcolor="#161b22",
                          height=350, margin=dict(l=10, r=10, t=30, b=10),
                          xaxis_title="Recall", yaxis_title="Precision",
                          xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
    st.plotly_chart(pr_fig, use_container_width=True)

    st.write("")
    m1, m2, m3, m4 = st.columns(4)
    for col, label, val in [(m1, "ROC-AUC", "0.998"), (m2, "PR-AUC", "0.909"),
                             (m3, "Precision @ 0.35", "30.0%"), (m4, "Recall @ 0.35", "96.5%")]:
        with col:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.write("")
    st.markdown("**Fraud Patterns Among Flagged Transactions**")
    st.caption("From the model's actual flagged (predicted-fraud) transactions in the test set")

    pat_col, buck_col = st.columns(2)

    with pat_col:
        st.markdown("Top category + hour combinations")
        pattern_data = pd.DataFrame([
            ("personal_care", 22, 324), ("personal_care", 23, 283),
            ("kids_pets", 23, 269), ("kids_pets", 22, 247),
            ("shopping_net", 23, 234), ("shopping_net", 22, 219),
            ("health_fitness", 22, 193), ("entertainment", 23, 172),
            ("health_fitness", 23, 170), ("travel", 23, 147),
        ], columns=["category", "hour", "count"])
        pattern_data["label"] = pattern_data["category"] + " @ " + pattern_data["hour"].astype(str) + ":00"
        pat_fig = px.bar(pattern_data.sort_values("count"), x="count", y="label", orientation="h",
                          color="count", color_continuous_scale="Reds")
        pat_fig.update_layout(template="plotly_dark", plot_bgcolor="#161b22", paper_bgcolor="#161b22",
                               height=340, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                               coloraxis_showscale=False, xaxis_title="Flagged transactions", yaxis_title=None)
        st.plotly_chart(pat_fig, use_container_width=True)
        st.caption("Flagged activity concentrates heavily in the 22:00-23:00 window across categories")

    with buck_col:
        st.markdown("Spend-anomaly severity of flagged transactions")
        bucket_data = pd.DataFrame({
            "bucket": ["Mild (z: -5 to 1)", "Very high (z: 5+)", "Elevated (z: 1-2)", "High (z: 3-5)", "Notable (z: 2-3)"],
            "count": [4178, 1237, 621, 548, 301],
        })
        buck_fig = px.pie(bucket_data, names="bucket", values="count", hole=0.5,
                           color_discrete_sequence=px.colors.sequential.Reds_r)
        buck_fig.update_layout(template="plotly_dark", plot_bgcolor="#161b22", paper_bgcolor="#161b22",
                                height=340, margin=dict(l=10, r=10, t=10, b=10),
                                legend=dict(font=dict(size=10)))
        st.plotly_chart(buck_fig, use_container_width=True)
        st.caption("~55% of flagged transactions have only mild spend anomaly — flagged on other signals (time, velocity), not amount alone")