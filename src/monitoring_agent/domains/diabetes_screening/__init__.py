"""BRFSS diabetes-screening preprocessing compatibility."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _binary(frame: pd.DataFrame, column: str) -> pd.Series:
    return _numeric(frame, column).fillna(0).clip(0, 1)


class BRFSSFeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless source-equivalent BRFSS feature engineering for inference."""

    def fit(self, x: pd.DataFrame, y: object = None) -> BRFSSFeatureEngineer:
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        frame = x.copy()
        bmi = _numeric(frame, "BMI")
        age = _numeric(frame, "Age")
        high_bp = _binary(frame, "HighBP")
        high_chol = _binary(frame, "HighChol")
        heart = _binary(frame, "HeartDiseaseorAttack")
        stroke = _binary(frame, "Stroke")
        diff_walk = _binary(frame, "DiffWalk")
        activity = _binary(frame, "PhysActivity")
        fruits = _binary(frame, "Fruits")
        veggies = _binary(frame, "Veggies")
        smoker = _binary(frame, "Smoker")
        heavy_alcohol = _binary(frame, "HvyAlcoholConsump")
        healthcare = _binary(frame, "AnyHealthcare")
        no_doc_cost = _binary(frame, "NoDocbcCost")
        chol_check = _binary(frame, "CholCheck")
        general_health = _numeric(frame, "GenHlth")
        mental_days = _numeric(frame, "MentHlth")
        physical_days = _numeric(frame, "PhysHlth")
        education = _numeric(frame, "Education")
        income = _numeric(frame, "Income")

        frame["bmi_category"] = np.select(
            [bmi.lt(18.5), bmi.lt(25), bmi.lt(30)],
            [0, 1, 2],
            default=3,
        ).astype(float)
        frame.loc[bmi.isna(), "bmi_category"] = np.nan
        frame["obese_flag"] = bmi.ge(30).astype(float)
        frame["overweight_or_obese_flag"] = bmi.ge(25).astype(float)
        frame["bmi_age_interaction"] = bmi * age
        frame["bmi_highbp_interaction"] = bmi * high_bp
        frame["bmi_highchol_interaction"] = bmi * high_chol
        frame["cardiometabolic_count"] = pd.concat(
            [high_bp, high_chol, heart, stroke, diff_walk, frame["obese_flag"]],
            axis=1,
        ).sum(axis=1)
        frame["bp_cholesterol_combo"] = high_bp * high_chol
        frame["cardio_event_history"] = ((heart + stroke) > 0).astype(float)
        inactivity = 1 - activity
        unhealthy_diet = ((fruits == 0) | (veggies == 0)).astype(float)
        frame["unhealthy_lifestyle_count"] = pd.concat(
            [smoker, inactivity, 1 - fruits, 1 - veggies, heavy_alcohol],
            axis=1,
        ).sum(axis=1)
        frame["healthy_diet_flag"] = (fruits * veggies).astype(float)
        frame["physical_inactivity_flag"] = inactivity.astype(float)
        frame["smoking_inactivity_combo"] = smoker * inactivity
        frame["diet_inactivity_combo"] = unhealthy_diet * inactivity
        no_healthcare = 1 - healthcare
        frame["healthcare_access_barrier"] = (
            (no_healthcare + no_doc_cost) > 0
        ).astype(float)
        frame["preventive_screening_gap"] = (
            ((1 - chol_check) + no_healthcare) > 0
        ).astype(float)
        frame["cholcheck_with_highchol_flag"] = chol_check * high_chol
        frame["poor_general_health_flag"] = general_health.ge(4).astype(float)
        frame["high_mental_distress_flag"] = mental_days.ge(14).astype(float)
        frame["high_physical_distress_flag"] = physical_days.ge(14).astype(float)
        frame["total_unhealthy_days"] = mental_days + physical_days
        frame["limited_functioning_flag"] = (
            (diff_walk + frame["high_physical_distress_flag"]) > 0
        ).astype(float)
        frame["health_burden_score"] = (
            general_health
            + mental_days / 30
            + physical_days / 30
            + diff_walk
        )
        frame["low_income_flag"] = income.le(3).astype(float)
        frame["low_education_flag"] = education.le(3).astype(float)
        frame["socioeconomic_risk_count"] = (
            frame["low_income_flag"] + frame["low_education_flag"]
        )
        frame["income_education_interaction"] = income * education
        frame["age_band"] = np.select(
            [age.le(4), age.le(8), age.le(11)],
            [0, 1, 2],
            default=3,
        ).astype(float)
        frame.loc[age.isna(), "age_band"] = np.nan
        frame["older_adult_flag"] = age.ge(10).astype(float)
        frame["age_bmi_interaction"] = age * bmi
        frame["age_cardiometabolic_interaction"] = (
            age * frame["cardiometabolic_count"]
        )
        return frame

    def get_feature_names_out(
        self,
        input_features: Iterable[str] | None = None,
    ) -> np.ndarray:
        if input_features is None:
            raise ValueError("input_features are required")
        sample = pd.DataFrame(columns=list(input_features))
        return self.transform(sample).columns.to_numpy(dtype=object)


__all__ = ["BRFSSFeatureEngineer"]
