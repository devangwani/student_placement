import json, joblib, numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Student Placement Predictor", layout="wide")
st.title("🎓 Student Placement Prediction & Report Card")

with open("config.json") as f:
    cfg = json.load(f)

features_found = [c for c in cfg["features_found"] if c is not None]
feature_labels = {v: k for k, v in cfg["feature_map"].items() if v}

pre_path = "models/preprocessor.joblib"
clf_path = "models/classifier.joblib"
thr_path = "models/threshold.json"
cohort_path = "models/cohort_medians.json"

missing = [p for p in [pre_path, clf_path, thr_path] if not os.path.exists(p)]
if missing:
    st.warning("Model artifacts not found. Please run **python src/train.py** first. Missing: " + ", ".join(missing))

if all(os.path.exists(p) for p in [pre_path, clf_path, thr_path]):
    pre = joblib.load(pre_path)
    clf = joblib.load(clf_path)
    thr = json.load(open(thr_path))["threshold"]
    cohort = json.load(open(cohort_path)) if os.path.exists(cohort_path) else None
else:
    pre = clf = thr = cohort = None

with st.form("input_form"):
    cols = st.columns(3)
    inputs = {}
    for i, col in enumerate(features_found):
        label = feature_labels.get(col, col)
        with cols[i % 3]:
            lname = label.lower()
            if "training" in lname:
                val = st.selectbox(f"{label}", ["No", "Yes"])
                inputs[col] = 1 if val == "Yes" else 0
            elif "score" in lname or "ssc" in lname or "hsc" in lname:
                inputs[col] = st.number_input(f"{label}", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
            elif "cgpa" in lname or "soft" in lname:
                # Allow decimal for CGPA and soft skills rating
                inputs[col] = st.number_input(f"{label}", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
            elif any(k in lname for k in ["internship", "project", "workshop", "certification"]):
                # Integer-only fields
                inputs[col] = st.number_input(f"{label}", min_value=0, value=0, step=1, format="%d")
            else:
                # Generic numeric fallback
                inputs[col] = st.number_input(f"{label}", min_value=0.0, value=0.0, step=1.0)


    submitted = st.form_submit_button("Predict")

if submitted:
    Xdf = pd.DataFrame([inputs])
    st.subheader("Input Summary")
    st.dataframe(Xdf)

    if pre is None or clf is None or thr is None:
        st.error("Model not ready. Please train the model first.")
        st.stop()

    Xt = pre.transform(Xdf)
    proba = float(clf.predict_proba(Xt)[:,1][0])
    pred = "Placed" if proba >= thr else "Not Placed"

    st.markdown(f"## Prediction: **{pred}**  (probability = {proba:.2f})")

    fig_g = go.Figure(go.Indicator(mode="gauge+number", value=proba*100,
                                   title={"text":"Placement Probability (%)"},
                                   gauge={"axis":{"range":[0,100]}}))
    st.plotly_chart(fig_g, use_container_width=True)

    radar_cols = [c for c in features_found if any(k in c.lower() for k in ["cgpa","aptitude","soft","project","intern","workshop","cert"])]
    if cohort and len(radar_cols) >= 3:
        student_vals = [float(Xdf[c]) for c in radar_cols]
        cohort_vals = [float(cohort.get(c, np.nan)) for c in radar_cols]
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(r=student_vals, theta=radar_cols, fill='toself', name='Student'))
        fig_r.add_trace(go.Scatterpolar(r=cohort_vals, theta=radar_cols, fill='toself', name='Placed Cohort Median'))
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, title="Skill/Readiness Radar")
        st.plotly_chart(fig_r, use_container_width=True)

    st.subheader("Quick Advice")
    tips = []

    def get(col_key):
        return inputs.get(cfg["feature_map"].get(col_key), None)

    cgpa = get("CGPA")
    apt = get("AptitudeTestScore") or get("ApptitudeTestScore")
    soft = get("SoftSkillRating") or get("SoftSkillrating")
    proj = get("Projects")
    intern = get("Internships")
    work = get("WorkshopsCertifications")
    train = get("PlacementTraining")

    if cgpa is not None and cgpa < 7.5: tips.append("Improve CGPA via targeted grade recovery or advanced coursework.")
    if apt is not None and apt < 70: tips.append("Daily aptitude practice (quant + logic) to target 70+.")
    if soft is not None and soft < 7: tips.append("Join speaking clubs/mock interviews to push soft skills to 7–8.")
    if proj is not None and proj < 2: tips.append("Add 1–2 industry-relevant projects with measurable outcomes.")
    if intern is not None and intern < 1: tips.append("Pursue at least one internship (even short/virtual).")
    if train is not None and int(train) == 0: tips.append("Complete placement training — high-leverage improvement.")
    if work is not None and work < 2: tips.append("Add 1–2 certifications aligned with your target roles.")
    if not tips:
        tips = ["You’re at or above placed cohort medians — focus on company-specific prep and mock interviews."]
    for t in tips: st.markdown(f"- {t}")

    st.caption("Note: This is a decision-support tool; outcomes depend on many external factors.")
