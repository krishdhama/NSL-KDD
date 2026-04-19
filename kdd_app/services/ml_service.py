import math
import os

import joblib
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODELS_DIR, "model.pkl"))
model.n_jobs = 1
columns = list(joblib.load(os.path.join(MODELS_DIR, "columns.pkl")))
top_services = joblib.load(os.path.join(MODELS_DIR, "top_services.pkl"))
feature_importance_map = dict(zip(columns, model.feature_importances_))


def get_top_services():
    return top_services


def format_feature_name(name):
    return name.replace("_", " ").title()


def build_model_input(form_data):
    input_dict = {col: 0 for col in columns}

    for key, value in form_data.items():
        if key in {"protocol_type", "flag", "service"}:
            continue
        if key in input_dict and value != "":
            input_dict[key] = float(value)

    protocol = form_data.get("protocol_type")
    if protocol:
        col = f"protocol_{protocol}"
        if col in input_dict:
            input_dict[col] = 1

    flag = form_data.get("flag")
    if flag:
        col = f"flag_{flag}"
        if col in input_dict:
            input_dict[col] = 1

    service = form_data.get("service")
    if service:
        if service not in top_services:
            service = "other"
        col = f"service_{service}"
        if col in input_dict:
            input_dict[col] = 1

    return input_dict


def get_top_responsible_features(input_dict, limit=5):
    active_features = []

    for name, value in input_dict.items():
        if not value:
            continue

        importance = float(feature_importance_map.get(name, 0.0))
        score = importance if value in (0, 1) else importance * (1 + math.log1p(abs(value)))
        active_features.append(
            {
                "column": name,
                "name": format_feature_name(name),
                "value": value,
                "importance": importance,
                "score": score,
            }
        )

    active_features.sort(key=lambda item: item["score"], reverse=True)
    return active_features[:limit]


def run_prediction(form_data):
    input_dict = build_model_input(form_data)
    df = pd.DataFrame([input_dict])

    pred = model.predict(df)[0]
    pred_str = str(pred).strip()

    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(df)[0].max())

    responsible_features = get_top_responsible_features(input_dict)
    model_inputs_used = {item["name"]: item["value"] for item in responsible_features}

    explanation = None
    if confidence is not None:
        try:
            from chat.genai import explain_prediction

            explanation = explain_prediction(
                prediction=pred_str,
                confidence=confidence,
                top_features=responsible_features,
                input_values=model_inputs_used,
            )
        except Exception as exc:
            explanation = (
                "AI explanation is unavailable right now. "
                f"Top likely columns were still identified from the model: {exc}"
            )

    if pred_str.lower() == "normal":
        result = "Threat Scan Result: Normal"
    else:
        result = f"Threat Scan Result: {pred_str.upper()}"

    return {
        "result": result,
        "prediction_explanation": explanation,
        "responsible_features": responsible_features,
        "prediction_confidence": confidence,
    }
