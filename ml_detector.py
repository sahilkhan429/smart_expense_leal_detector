
import pandas as pd
from sklearn.ensemble import IsolationForest


REQUIRED_COLUMNS = [
    "Date",
    "Description",
    "Category",
    "Amount",
]


def detect_ml_anomalies(df):
    """
    Flag potential expense anomalies using IsolationForest
    and a robust statistical check.

    Returns flagged rows with an explanation of the detection.
    An anomaly is not proof of fraud or unnecessary spending.
    """

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df.empty:
        return df.iloc[0:0].copy()

    result = df.copy()
    result["Amount"] = pd.to_numeric(
        result["Amount"], errors="coerce"
    )

    valid_mask = (
        result["Amount"].notna()
        & (result["Amount"] >= 0)
    )
    valid = result.loc[valid_mask].copy()

    # Very small or constant datasets are not sufficient
    # for meaningful anomaly detection.
    if len(valid) < 5 or valid["Amount"].nunique() < 2:
        return result.iloc[0:0].copy()

    amounts = valid["Amount"].astype(float)

    # Robust outlier score based on median absolute deviation.
    median_amount = amounts.median()
    absolute_deviations = (amounts - median_amount).abs()
    mad = absolute_deviations.median()

    if mad > 0:
        robust_score = (
            0.6745 * (amounts - median_amount).abs() / mad
        )
        robust_outlier = robust_score >= 3.5
    else:
        # If most values are identical, use the non-zero
        # deviations from the median as a fallback.
        robust_outlier = amounts > median_amount * 3

    # IsolationForest is used as a second signal.
    model = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=42,
    )

    predictions = model.fit_predict(
        amounts.to_numpy().reshape(-1, 1)
    )

    isolation_flag = pd.Series(
        predictions == -1,
        index=valid.index
    )

    # Require support from the robust statistical check
    # to reduce isolated false alarms on small datasets.
    flagged = robust_outlier & isolation_flag

    anomalies = valid.loc[flagged].copy()
    anomalies["ML Detection"] = "Potential anomaly"
    anomalies["Detection Reason"] = (
        "Flagged by IsolationForest and robust amount check"
    )

    return anomalies.reset_index(drop=True)
