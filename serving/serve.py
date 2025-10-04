import os
import logging
import traceback
from typing import Any, Dict, List

import joblib
import pandas as pd
from flask import Flask, request, jsonify

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Paths (match training/data defaults) - updated for shared volume
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/shared/models/gbr.joblib")
PREPROC_PATH = os.environ.get("PREPROC_PATH", "/app/shared/models/preprocessor.joblib")

app = Flask(__name__)

def _load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    saved = joblib.load(model_path)
    model = saved.get("model") if isinstance(saved, dict) and "model" in saved else saved
    return model

def _load_preprocessor(preproc_path: str):
    if not os.path.exists(preproc_path):
        raise FileNotFoundError(f"Preprocessor file not found: {preproc_path}")
    saved = joblib.load(preproc_path)
    # saved may be {"pipeline": ..., "cols": ...} or a raw pipeline
    pipeline = saved.get("pipeline") if isinstance(saved, dict) and "pipeline" in saved else saved
    # try to recover expected input feature names
    feature_names = None
    feature_names = getattr(pipeline, "feature_names_in_", None)
    if feature_names is None:
        # try transformers metadata
        try:
            parts = []
            for _, _, cols in getattr(pipeline, "transformers_", []):
                if isinstance(cols, (list, tuple)):
                    parts.extend(list(cols))
            if parts:
                feature_names = parts
        except Exception:
            feature_names = None
    return pipeline, feature_names

# load once at startup
try:
    MODEL = _load_model(MODEL_PATH)
    PREPROC, PREPROC_FEATURES = _load_preprocessor(PREPROC_PATH)
    LOG.info("Loaded model from %s and preprocessor from %s", MODEL_PATH, PREPROC_PATH)
except Exception:
    MODEL = None
    PREPROC = None
    PREPROC_FEATURES = None
    LOG.warning("Failed to load model/preprocessor on startup:\n%s", traceback.format_exc())

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok" if MODEL is not None and PREPROC is not None else "uninitialized",
        "model_loaded": MODEL is not None,
        "preprocessor_loaded": PREPROC is not None,
        "model_path": MODEL_PATH,
        "preproc_path": PREPROC_PATH,
    })

def _ensure_df(payload: Any) -> pd.DataFrame:
    # Accept single row as dict, or list of dicts, or {"data": {...}} wrapper
    if isinstance(payload, dict) and "data" in payload:
        payload = payload["data"]
    if isinstance(payload, dict):
        # single row
        df = pd.DataFrame([payload])
    elif isinstance(payload, list):
        df = pd.DataFrame(payload)
    else:
        raise ValueError("Unsupported JSON payload. Send a dict (one row) or list of dicts.")
    return df

@app.route("/predict", methods=["POST"])
def predict():
    if MODEL is None or PREPROC is None:
        return jsonify({"error": "model or preprocessor not loaded on server"}), 503

    try:
        payload = request.get_json(force=True)
        df = _ensure_df(payload)

        # determine expected columns
        expected = None
        if PREPROC_FEATURES is not None:
            expected = list(PREPROC_FEATURES)
        else:
            # try saved metadata if present
            saved = joblib.load(PREPROC_PATH)
            expected = saved.get("cols") if isinstance(saved, dict) else None

        if expected is not None:
            missing = [c for c in expected if c not in df.columns]
            if missing:
                return jsonify({"error": "missing input columns", "missing": missing, "expected": expected}), 400
            # reorder to original expectation to avoid ambiguity
            try:
                df = df[expected]
            except Exception:
                pass

        # Apply preprocessor (it expects DataFrame with original columns)
        X_trans = PREPROC.transform(df)
        preds = MODEL.predict(X_trans)
        preds_list = [float(x) for x in preds]

        return jsonify({"predictions": preds_list, "n": len(preds_list)})
    except Exception as exc:
        LOG.error("Prediction error: %s", traceback.format_exc())
        return jsonify({"error": "prediction failed", "detail": str(exc)}), 500

if __name__ == "__main__":
    # Use 0.0.0.0 for local network access; change port as needed.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)