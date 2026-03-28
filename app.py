from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)
# hlo

# ---------------- LOAD FILES ----------------
import joblib
import os

# Point to the new 'models/' directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODELS_DIR, "model.pkl"))
columns = joblib.load(os.path.join(MODELS_DIR, "columns.pkl"))
top_services = joblib.load(os.path.join(MODELS_DIR, "top_services.pkl"))

# ---------------- LABEL MAPPING ----------------
label_map = {
    0: "Normal ✅",
    1: "DoS 🚨",
    2: "Probe ⚠️",
    3: "U2R 🔥",
    4: "R2L ⚡"
}

# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("index2.html", services=top_services)


# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("datget")
        form_data = request.form.to_dict()

        # ----------- CREATE FULL INPUT -----------
        input_dict = {col: 0 for col in columns}
        print("dict created")
        # ----------- NUMERIC FEATURES -----------
        numeric_features = [
            "duration", "src_bytes", "dst_bytes",
            "count", "srv_count"
        ]

        for key in numeric_features:
            if key in form_data and form_data[key] != "":
                input_dict[key] = float(form_data[key])

        # ----------- PROTOCOL ENCODING -----------
        protocol = form_data.get("protocol_type")
        if protocol:
            col = f"protocol_{protocol}"
            if col in input_dict:
                input_dict[col] = 1
                print("encoded 1 col")

        # ----------- FLAG ENCODING -----------
        flag = form_data.get("flag")
        if flag:
            col = f"flag_{flag}"
            if col in input_dict:
                input_dict[col] = 1
                print("encoded 2 col")

        # ----------- SERVICE (TOP-K + OTHER) -----------
        service = form_data.get("service")
        if service:
            if service not in top_services:
                service = "other"

            col = f"service_{service}"
            if col in input_dict:
                input_dict[col] = 1
                print("encoded 3 col")

        # ----------- DATAFRAME -----------
        df = pd.DataFrame([input_dict])
        print("df_created")
        # ----------- PREDICTION -----------
        pred = model.predict(df)[0]
        print("pred_made:", pred)
        
        # Format the raw string returned by the model
        pred_str = str(pred).strip().lower()
        if pred_str == "normal":
            result = f"🔥 RESULT: Normal ✅"
        else:
            result = f"🔥 RESULT: {pred_str.upper()} 🚨"

        return render_template("index2.html", result=result, services=top_services)

    except Exception as e:
        return str(e)


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)