# src/predict.py
"""
Load models/baseline.pkl, build window features from a dataset,
and print predictions (class + probabilities) for a few sample windows.
Run as:
    python -m src.predict
"""
from pathlib import Path
import joblib
import pandas as pd
from src.features import load_data, make_window_features

MODEL_PATH = Path("models") / "baseline.pkl"
DATA_PATH = Path("data") / "dataset_forced_failures.csv"

def main(data_path: str = str(DATA_PATH)):
    print("Loading model from:", MODEL_PATH)
    pipe = joblib.load(MODEL_PATH)

    print("Loading data from:", data_path)
    df = load_data(data_path)

    print("Making window features...")
    X, y, meta = make_window_features(df, window_sec=60, step_sec=30)
    print("Feature matrix shape:", X.shape)

    if X.shape[0] == 0:
        print("No windows found. Exiting.")
        return

    # Predict probabilities and labels
    probs = pipe.predict_proba(X)
    preds = pipe.predict(X)

    # We assume classes are [0,1,2] but get actual classes
    classes = pipe.named_steps['rf'].classes_ if hasattr(pipe.named_steps['rf'], 'classes_') else None
    print("Model classes:", classes)

    # Add predictions into meta and show top risky windows (highest prob of class 2)
    df_out = meta.copy()
    df_out = df_out.reset_index(drop=True)
    df_out['pred'] = preds
    # find index of class '2' in classes
    if classes is not None and 2 in classes:
        idx2 = list(classes).index(2)
        df_out['prob_fail'] = [p[idx2] for p in probs]
    else:
        df_out['prob_fail'] = 0.0

    # show top 8 highest failure risk windows
    df_top = df_out.sort_values('prob_fail', ascending=False).head(8)
    print("\\nTop predicted-risk windows:")
    print(df_top.to_string(index=False))

    # Save predictions to file
    outp = Path("models") / "predictions_sample.csv"
    df_out.to_csv(outp, index=False)
    print(f"Saved predictions to: {outp}")

if __name__ == "__main__":
    main()
