from pathlib import Path

import joblib
import numpy as np
import streamlit as st


RED_COLOR = "#e53935"
GREEN_COLOR = "#2e7d32"


def render_numeric_slider(feature_name, bounds, defaults, descriptions):
    cfg = bounds[feature_name]
    is_int = bool(cfg.get("is_int", True))
    min_val = cfg["min"]
    max_val = cfg["max"]
    default_val = defaults.get(feature_name, (min_val + max_val) / 2)
    step_val = cfg.get("step", 1)

    if is_int:
        value = st.slider(
            feature_name,
            min_value=int(min_val),
            max_value=int(max_val),
            value=int(default_val),
            step=int(step_val),
            help=descriptions.get(feature_name, ""),
        )
    else:
        value = st.slider(
            feature_name,
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default_val),
            step=float(step_val),
            help=descriptions.get(feature_name, ""),
        )

    st.caption(descriptions.get(feature_name, ""))
    return value


def render_categorical_select(feature_name, options, descriptions):
    value = st.selectbox(
        feature_name,
        options=options,
        help=descriptions.get(feature_name, ""),
    )
    st.caption(descriptions.get(feature_name, ""))
    return value


st.set_page_config(page_title="Passenger Satisfaction Predictor", layout="centered")
st.title("Passenger Satisfaction Predictor")
st.write("Fill the passenger details, then click Predict Satisfaction.")

model_path = Path(__file__).with_name("best_model.joblib")
if not model_path.exists():
    st.error("Model file not found. Please regenerate best_model.joblib first.")
    st.stop()

artifact = joblib.load(model_path)

required_keys = [
    "model",
    "scaler",
    "feature_order",
    "categorical_features",
    "feature_descriptions",
    "feature_bounds",
    "feature_defaults",
    "categorical_options",
    "category_to_code",
]
missing_keys = [k for k in required_keys if k not in artifact]
if missing_keys:
    st.error(
        "Artifact format is old. Missing keys: "
        + ", ".join(missing_keys)
        + ". Rebuild best_model.joblib."
    )
    st.stop()

model = artifact["model"]
scaler = artifact["scaler"]
feature_order = artifact["feature_order"]
categorical_features = set(artifact["categorical_features"])
descriptions = artifact["feature_descriptions"]
bounds = artifact["feature_bounds"]
defaults = artifact["feature_defaults"]
categorical_options = artifact["categorical_options"]
category_to_code = artifact["category_to_code"]
label_map = artifact.get("label_map", {0: "Class 0", 1: "Class 1"})
demo_test_cases = artifact.get("demo_test_cases", {})

st.caption(f"Model: {artifact.get('model_name', 'Trained Classifier')}")

with st.form("satisfaction_form"):
    inputs = {}
    left_col, right_col = st.columns(2)

    for idx, feature in enumerate(feature_order):
        current_col = left_col if idx % 2 == 0 else right_col
        with current_col:
            if feature in categorical_features:
                inputs[feature] = render_categorical_select(
                    feature,
                    categorical_options[feature],
                    descriptions,
                )
            else:
                inputs[feature] = render_numeric_slider(
                    feature,
                    bounds,
                    defaults,
                    descriptions,
                )

    predict_clicked = st.form_submit_button("Predict Satisfaction", use_container_width=True)

if predict_clicked:
    encoded_row = {}

    for feature in feature_order:
        raw_value = inputs[feature]
        if feature in categorical_features:
            encoded_row[feature] = category_to_code[feature][raw_value]
        else:
            encoded_row[feature] = float(raw_value)

    row = np.array([[encoded_row[f] for f in feature_order]], dtype=float)
    row_scaled = scaler.transform(row)

    pred = int(model.predict(row_scaled)[0])
    proba = model.predict_proba(row_scaled)[0]

    predicted_label = label_map.get(pred, f"Class {pred}")
    predicted_probability = float(proba[pred]) * 100

    if pred == 0:
        status_color = RED_COLOR
    else:
        status_color = GREEN_COLOR

    st.markdown("### Predicted class:")
    st.markdown(
        f"""
        <div style=\"padding:12px;border-radius:10px;background:{status_color};color:white;font-weight:700;\">
            {predicted_label}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Prediction probability:")
    st.markdown(f"## {predicted_probability:.2f}%")
    st.markdown(
        f"""
        <div style=\"width:100%;background:#eceff1;border-radius:999px;height:16px;overflow:hidden;\">
            <div style=\"width:{predicted_probability:.2f}%;background:{status_color};height:16px;\"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if demo_test_cases:
    with st.expander("Quick test cases"):
        st.write("Dissatisfied test case")
        st.json(demo_test_cases.get("Dissatisfied", {}))
        st.write("Satisfied test case")
        st.json(demo_test_cases.get("Satisfied", {}))
