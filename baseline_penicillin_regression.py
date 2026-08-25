"""
Baseline Regression Model for Penicillin Concentration
=======================================================
Uses normalized batch time (0-1) as the sole input feature
with Polynomial Regression (degree 3) via scikit-learn.

Train: Regimes 1-3 | Test: Regime 4 (fault regime)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error


# ---------------------------------------------------------------------------
# 1. Data preparation
# ---------------------------------------------------------------------------

def normalize_time_per_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize time to [0, 1] within each batch (% batch completion)."""
    df = df.copy()
    df["time_norm"] = df.groupby("batch_id")["time"].transform(
        lambda t: (t - t.min()) / (t.max() - t.min())
    )
    return df


# ---------------------------------------------------------------------------
# 2. Model construction
# ---------------------------------------------------------------------------

def build_polynomial_model(degree: int = 3):
    """Return a scikit-learn pipeline: PolynomialFeatures -> LinearRegression."""
    return make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression(),
    )


# ---------------------------------------------------------------------------
# 3. Training
# ---------------------------------------------------------------------------

def train_model(df_train: pd.DataFrame, degree: int = 3):
    """Normalize time, fit polynomial model, return (model, prepared df)."""
    df_train = normalize_time_per_batch(df_train)

    X_train = df_train[["time_norm"]].values
    y_train = df_train["penicillin"].values

    model = build_polynomial_model(degree=degree)
    model.fit(X_train, y_train)

    return model, df_train


# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, df_test: pd.DataFrame) -> float:
    """Predict on test set, compute and return RMSE in g/L."""
    df_test = normalize_time_per_batch(df_test)

    X_test = df_test[["time_norm"]].values
    y_test = df_test["penicillin"].values

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse, df_test, y_pred


# ---------------------------------------------------------------------------
# 5. Main entry point
# ---------------------------------------------------------------------------

def main(df_train: pd.DataFrame, df_test: pd.DataFrame, degree: int = 3):
    """Full pipeline: train on regimes 1-3, evaluate on regime 4."""

    print(f"Training batches : {df_train['batch_id'].nunique()}")
    print(f"Test batches     : {df_test['batch_id'].nunique()}")
    print(f"Polynomial degree: {degree}")
    print("-" * 50)

    # Train
    model, df_train_prep = train_model(df_train, degree=degree)

    # Evaluate
    rmse, df_test_prep, y_pred = evaluate_model(model, df_test)

    print(f"Test RMSE: {rmse:.4f} g/L")
    print("-" * 50)

    # Report polynomial coefficients for interpretability
    poly_features = model.named_steps["polynomialfeatures"]
    lin_reg = model.named_steps["linearregression"]
    feature_names = poly_features.get_feature_names_out(["t"])
    print("Model coefficients:")
    for name, coef in zip(feature_names, lin_reg.coef_):
        print(f"  {name:>6s}: {coef:+.6f}")
    print(f"  {'bias':>6s}: {lin_reg.intercept_:+.6f}")

    return model, rmse, y_pred


# ---------------------------------------------------------------------------
# 6. Example usage with synthetic data (replace with real DataFrames)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Synthetic data for demonstration (replace with actual data) --------
    rng = np.random.default_rng(42)

    def _make_batch(batch_id, n_points=100, fault=False):
        t = np.linspace(0, 400, n_points)  # minutes
        # Simulated penicillin growth curve (sigmoid-like)
        p = 1.5 / (1 + np.exp(-0.02 * (t - 200)))
        if fault:
            p *= 0.7  # reduced yield under fault
        p += rng.normal(0, 0.05, size=n_points)
        return pd.DataFrame({
            "batch_id": batch_id,
            "time": t,
            "penicillin": p,
        })

    # Regimes 1-3 (training)
    df_train = pd.concat(
        [_make_batch(f"batch_{i}") for i in range(1, 16)],
        ignore_index=True,
    )

    # Regime 4 (fault / test)
    df_test = pd.concat(
        [_make_batch(f"fault_{i}", fault=True) for i in range(1, 6)],
        ignore_index=True,
    )

    # Run
    model, rmse, y_pred = main(df_train, df_test, degree=3)
