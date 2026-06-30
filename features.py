"""Feature engineering shared by training and Streamlit inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create leakage-safe date and coupon features.

    The reference date is learned only from the training fold during fit.
    This lets the transformer work correctly inside cross-validation.
    """

    date_columns = ("signup_date", "last_purchase_date")

    def fit(self, X: pd.DataFrame, y=None):
        frame = self._as_frame(X)
        observed_dates: list[pd.Series] = []
        for column in self.date_columns:
            if column in frame.columns:
                observed_dates.append(pd.to_datetime(frame[column], errors="coerce"))

        if observed_dates:
            all_dates = pd.concat(observed_dates, ignore_index=True)
            max_date = all_dates.max()
        else:
            max_date = pd.NaT

        if pd.isna(max_date):
            max_date = pd.Timestamp("2025-01-01")

        self.reference_date_ = pd.Timestamp(max_date).normalize()
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "reference_date_"):
            raise RuntimeError("FeatureEngineer must be fitted before transform.")

        frame = self._as_frame(X).copy()

        signup = self._parse_date(frame, "signup_date")
        last_purchase = self._parse_date(frame, "last_purchase_date")

        frame["customer_tenure_days"] = (
            self.reference_date_ - signup
        ).dt.days.clip(lower=0)
        frame["days_since_last_purchase"] = (
            self.reference_date_ - last_purchase
        ).dt.days.clip(lower=0)
        frame["signup_year"] = signup.dt.year
        frame["signup_month"] = signup.dt.month
        frame["last_purchase_year"] = last_purchase.dt.year
        frame["last_purchase_month"] = last_purchase.dt.month

        if "coupon_code" in frame.columns:
            coupon = frame["coupon_code"]
            frame["is_coupon_used"] = (
                coupon.notna() & coupon.astype(str).str.strip().ne("")
            ).astype(int)
        else:
            frame["is_coupon_used"] = 0

        for column in ("total_spent", "lifetime_value"):
            if column in frame.columns:
                values = pd.to_numeric(frame[column], errors="coerce").clip(lower=0)
                frame[column] = np.log1p(values)

        columns_to_drop = [
            column
            for column in ("customer_id", "coupon_code", *self.date_columns)
            if column in frame.columns
        ]
        return frame.drop(columns=columns_to_drop)

    @staticmethod
    def _as_frame(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)

    @staticmethod
    def _parse_date(frame: pd.DataFrame, column: str) -> pd.Series:
        if column not in frame.columns:
            return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
        return pd.to_datetime(frame[column], errors="coerce")
