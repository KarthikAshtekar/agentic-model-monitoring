"""Build the canonical evidence-backed HTML explainer for this repository.

The script deliberately re-computes bundle, reference-data, metric, and deterministic
scenario evidence. It does not call the live LLM provider, regenerate scenarios, retrain
models, or overwrite historical monitoring/evaluation reports.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from monitoring_agent.bundle.validation import validate_bundle
from monitoring_agent.models.bundle import RegisteredModelBundle
from monitoring_agent.models.registry import ModelRegistry
from monitoring_agent.monitoring.data_quality import evaluate_data_quality
from monitoring_agent.monitoring.engine import MonitoringEngine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = PROJECT_ROOT / "project_explainer_artifact.json"
EVIDENCE_PATH = PROJECT_ROOT / "reports/project_explainer_evidence.json"
HTML_PATH = PROJECT_ROOT / "project_explainer.html"
ARCHIVE_ROOT = PROJECT_ROOT / "reports/explainer_archive"
SCENARIOS = (
    "normal_operation",
    "feature_drift",
    "data_quality_failure",
    "performance_degradation",
    "unlabelled_drift",
    "insufficient_labels",
)
MODEL_IDS = ("credit_default", "diabetes_risk")


def _read_json(relative_path: str) -> dict[str, Any]:
    return json.loads((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))


def _read_yaml(relative_path: str) -> dict[str, Any]:
    payload = yaml.safe_load((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a mapping in {relative_path}.")
    return payload


def _preservation_integrity() -> dict[str, Any]:
    """Check protected credit evidence without modifying or normalizing any file."""
    manifest_path = "reports/refactor/credit_preservation_manifest.json"
    manifest = _read_json(manifest_path)
    rows: list[dict[str, Any]] = []
    for entry in manifest["files"]:
        relative_path = entry["relative_path"]
        path = PROJECT_ROOT / relative_path
        exists = path.is_file()
        actual_size = path.stat().st_size if exists else None
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest() if exists else None
        rows.append(
            {
                "relative_path": relative_path,
                "exists": exists,
                "expected_size": entry["file_size"],
                "actual_size": actual_size,
                "size_matches": actual_size == entry["file_size"],
                "expected_sha256": entry["sha256"],
                "actual_sha256": actual_sha256,
                "sha256_matches": actual_sha256 == entry["sha256"],
            }
        )
    mismatches = [row for row in rows if not row["size_matches"] or not row["sha256_matches"]]
    return {
        "manifest": manifest_path,
        "total_files": len(rows),
        "matched_files": len(rows) - len(mismatches),
        "all_match": not mismatches,
        "mismatches": mismatches,
    }


def _git_value(*arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _run_test_suite() -> dict[str, Any]:
    """Run the full local suite and retain a compact, current QA receipt."""
    completed = subprocess.run(
        [sys.executable, "-m", "pytest"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    output = "\n".join(item for item in (completed.stdout, completed.stderr) if item)
    counts = {}
    for name in ("failed", "passed", "warnings", "skipped"):
        matches = re.findall(rf"(\d+) {name}", output)
        counts[name] = int(matches[-1]) if matches else 0
    failed_nodes = re.findall(r"^FAILED ([^\s]+)", output, flags=re.MULTILINE)
    summary_line = next(
        (
            line.strip()
            for line in reversed(output.splitlines())
            if " passed" in line or " failed" in line
        ),
        "pytest did not emit a parseable summary",
    )
    return {
        "command": f"{Path(sys.executable).name} -m pytest",
        "return_code": completed.returncode,
        "counts": counts,
        "failed_nodes": failed_nodes,
        "summary": summary_line,
    }


def _safe_warning(message: str) -> str:
    """Keep compatibility warnings useful without leaking machine-specific paths."""
    return re.sub(r"[A-Za-z]:\\[^\n]+", "[local path omitted]", message).strip()


def _threshold_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "sample_count": len(labels),
        "positive_count": int(labels.sum()),
        "predicted_positive_count": int(predictions.sum()),
        "predicted_positive_rate": float(predictions.mean()),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def _model_specification(bundle: RegisteredModelBundle) -> dict[str, Any]:
    artifact = bundle.load_inference_artifact()
    calibrator: Any | None = None
    base = artifact
    if hasattr(artifact, "base_estimator"):
        base = artifact.base_estimator
        calibrator = getattr(artifact, "calibration_model", None)
    steps = []
    final = base
    if hasattr(base, "steps"):
        steps = [{"name": name, "class": type(step).__name__} for name, step in base.steps]
        final = base.steps[-1][1]
    parameter_names = (
        "objective",
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "gamma",
        "reg_alpha",
        "reg_lambda",
        "scale_pos_weight",
        "random_state",
        "eval_metric",
        "n_jobs",
    )
    parameters: dict[str, Any] = {}
    if hasattr(final, "get_params"):
        all_parameters = final.get_params(deep=False)
        parameters = {
            name: all_parameters[name]
            for name in parameter_names
            if name in all_parameters and all_parameters[name] is not None
        }
    return {
        "artifact_class": f"{type(artifact).__module__}.{type(artifact).__name__}",
        "pipeline_steps": steps,
        "classifier_class": f"{type(final).__module__}.{type(final).__name__}",
        "classifier_parameters": parameters,
        "calibrator_class": (
            None
            if calibrator is None
            else f"{type(calibrator).__module__}.{type(calibrator).__name__}"
        ),
        "calibrator_parameters": (
            None
            if calibrator is None or not hasattr(calibrator, "get_params")
            else calibrator.get_params(deep=False)
        ),
    }


def _profile_reference(
    bundle: RegisteredModelBundle,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    registered = bundle.registered_manifest
    metadata = bundle.load_metadata()
    schema = bundle.load_feature_schema()
    features = bundle.load_reference_features()
    labels_frame = bundle.load_reference_labels()
    predictions = bundle.load_reference_predictions()
    identifier = registered.data_contract.identifier_column
    labels = labels_frame["actual_label"].to_numpy(dtype="int8")
    probabilities = predictions["predicted_probability"].to_numpy(dtype=float)
    numeric = features.select_dtypes(include=[np.number])
    profile = {
        "model_id": bundle.model_id,
        "display_name": registered.identity.display_name,
        "dataset": metadata.dataset_name,
        "split": metadata.reference_split,
        "unit": registered.business_context.prediction_unit,
        "target": registered.data_contract.target_column,
        "positive_outcome": registered.business_context.positive_outcome,
        "rows": len(features),
        "raw_feature_count": len(schema.ordered_features),
        "preprocessed_feature_count": registered.data_contract.preprocessed_feature_count,
        "duplicate_record_ids": int(features[identifier].duplicated().sum()),
        "exact_duplicate_rows": int(features.duplicated().sum()),
        "missing_cells": int(features.isna().sum().sum()),
        "infinite_numeric_cells": int(
            np.isinf(numeric.to_numpy(dtype=float)).sum() if len(numeric.columns) else 0
        ),
        "label_rows": len(labels_frame),
        "positive_count": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "probability_min": float(probabilities.min()),
        "probability_max": float(probabilities.max()),
        "default_threshold": metadata.default_threshold,
        "operating_threshold": metadata.operating_threshold,
    }
    metrics = {
        "default_0_50": _threshold_metrics(
            labels,
            probabilities,
            metadata.default_threshold,
        ),
        "operating_0_25": _threshold_metrics(
            labels,
            probabilities,
            metadata.operating_threshold,
        ),
    }
    dictionary = []
    categorical = set(schema.categorical_features)
    integer = set(schema.integer_constrained_features)
    for feature in schema.features:
        allowed = feature.allowed_values
        dictionary.append(
            {
                "model_id": bundle.model_id,
                "position": feature.position + 1,
                "field": feature.name,
                "kind": "categorical/discrete" if feature.name in categorical else "numeric",
                "dtype": feature.dtype,
                "integer_constrained": feature.name in integer,
                "nullable": feature.nullable,
                "observed_range": (
                    "-"
                    if feature.observed_min is None or feature.observed_max is None
                    else f"{feature.observed_min:g} to {feature.observed_max:g}"
                ),
                "allowed_values": (
                    "-" if allowed is None else ", ".join(str(value) for value in allowed)
                ),
                "meaning": _feature_meaning(bundle.model_id, feature.name),
            }
        )
    return profile, metrics, dictionary


def _feature_meaning(model_id: str, name: str) -> str:
    credit_meanings = {
        "LIMIT_BAL": "Granted credit limit.",
        "EDUCATION": "Encoded education category.",
        "MARRIAGE": "Encoded marital-status category.",
        "AGE": "Account-holder age in years.",
        "PAY_0": "Most recent repayment-status code.",
        "RecentPaymentDelay": "Engineered recent repayment-delay indicator.",
        "MaxPaymentDelay": "Maximum repayment-delay code across observed months.",
        "NumDelayedMonths": "Count of delayed-payment months.",
        "AvgBillAmount": "Mean bill amount across six months.",
        "AvgPaymentAmount": "Mean payment amount across six months.",
        "PaymentToLimitRatio": "Payment amount relative to the granted limit.",
        "AvgBillToLimitRatio": "Mean bill-to-limit utilization ratio.",
        "AvgPaymentToBillRatio": "Mean payment-to-bill ratio.",
    }
    diabetes_meanings = {
        "HighBP": "Self-reported high-blood-pressure indicator.",
        "HighChol": "Self-reported high-cholesterol indicator.",
        "CholCheck": "Cholesterol check within the survey-defined period.",
        "BMI": "Body-mass-index value.",
        "Smoker": "Survey smoking-history indicator.",
        "Stroke": "Self-reported stroke-history indicator.",
        "HeartDiseaseorAttack": "Self-reported coronary disease or heart-attack history.",
        "PhysActivity": "Recent physical-activity indicator.",
        "Fruits": "Daily fruit-consumption indicator.",
        "Veggies": "Daily vegetable-consumption indicator.",
        "HvyAlcoholConsump": "Heavy-alcohol-consumption indicator.",
        "AnyHealthcare": "Any health-care coverage indicator.",
        "NoDocbcCost": "Could not see a doctor because of cost.",
        "GenHlth": "Ordinal self-rated general health.",
        "MentHlth": "Poor mental-health days in the past 30 days.",
        "PhysHlth": "Poor physical-health days in the past 30 days.",
        "DiffWalk": "Serious difficulty walking or climbing stairs.",
        "Sex": "Encoded survey sex field.",
        "Age": "Ordinal age-band code.",
        "Education": "Ordinal education-level code.",
        "Income": "Ordinal income-band code.",
    }
    if model_id == "credit_default":
        if name in credit_meanings:
            return credit_meanings[name]
        if name.startswith("PAY_"):
            return "Monthly repayment-status code."
        if name.startswith("BILL_AMT"):
            return "Monthly bill statement amount."
        if name.startswith("PAY_AMT"):
            return "Monthly payment amount."
        if name.startswith("BillToLimitRatio"):
            return "Engineered monthly bill-to-limit utilization ratio."
    return diabetes_meanings.get(
        name, "Engineered or source predictor retained by the fitted contract."
    )


def _scenario_evidence(
    registry: ModelRegistry,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result_rows: list[dict[str, Any]] = []
    design_rows: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        engine = MonitoringEngine(model_id=model_id, registry=registry)
        scenario_config = _read_yaml(f"configs/scenarios/{model_id}.yaml")
        for scenario in SCENARIOS:
            result = engine.run_scenario(scenario)
            critical_ids = [
                item.evidence_id for item in result.evidence if item.status == "critical"
            ]
            operating = result.performance.metrics_at_operating_threshold
            result_rows.append(
                {
                    "model_id": model_id,
                    "scenario": scenario,
                    "batch_blocked": result.batch_blocked,
                    "performance_evaluated": result.performance.evaluated,
                    "label_coverage": result.label_coverage_rate,
                    "critical_features": result.drift.critical_feature_count,
                    "incident_candidates": ", ".join(result.incident_candidates),
                    "overall_severity": result.overall_severity,
                    "critical_evidence": ", ".join(critical_ids) or "none",
                    "operating_recall": operating.get("recall"),
                    "operating_precision": operating.get("precision"),
                    "operating_f1": operating.get("f1"),
                    "roc_auc": operating.get("roc_auc"),
                    "pr_auc": operating.get("pr_auc"),
                    "brier_score": operating.get("brier_score"),
                    "run_id": result.run_id,
                }
            )
            configured = scenario_config["scenarios"][scenario]
            parameters = configured.get("parameters", {})
            design_rows.append(
                {
                    "model_id": model_id,
                    "scenario": scenario,
                    "simulation_type": configured["simulation_type"],
                    "sample_size": scenario_config["defaults"]["sample_size"],
                    "seed": scenario_config["defaults"]["random_seed"],
                    "labels": (
                        "none"
                        if scenario == "unlabelled_drift"
                        else str(parameters.get("labelled_sample_count", "all 1,000"))
                    ),
                    "configured_change": _describe_scenario_change(scenario, parameters),
                    "expected_candidates": ", ".join(configured["expected_incident_candidates"]),
                }
            )
    return result_rows, design_rows


def _describe_scenario_change(scenario: str, parameters: dict[str, Any]) -> str:
    if scenario == "normal_operation":
        return "Unchanged stratified replay sample."
    if scenario in {"feature_drift", "unlabelled_drift"}:
        shifts = parameters.get("discrete_shifts")
        discrete = (
            ", ".join(str(item["feature"]) for item in shifts)
            if shifts
            else f"{parameters.get('discrete_feature')} + {parameters.get('linked_feature')}"
        )
        return (
            f"Scale {parameters.get('continuous_feature')} by "
            f"{parameters.get('continuous_scale_factor')}; shift {discrete}."
        )
    if scenario == "data_quality_failure":
        return (
            f"Duplicate {parameters.get('duplicate_record_count')} IDs; set "
            f"{parameters.get('missing_fraction', 0):.0%} of "
            f"{parameters.get('missing_feature')} missing; inject observed-range violations "
            f"in {parameters.get('range_violation_feature')}."
        )
    if scenario == "performance_degradation":
        return (
            f"Flip {parameters.get('negative_to_positive_count')} lowest-score negative "
            "labels to positive."
        )
    return f"Retain {parameters.get('labelled_sample_count')} aligned labels."


def _module_inventory() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    modules: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    for base in (PROJECT_ROOT / "src/monitoring_agent", PROJECT_ROOT / "scripts"):
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            objects = [
                node.name
                for node in tree.body
                if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
                and not node.name.startswith("_")
            ]
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            modules.append(
                {
                    "path": relative,
                    "stage": _module_stage(relative),
                    "purpose": (ast.get_docstring(tree) or "Package wiring.").strip(),
                    "key_objects": ", ".join(objects) or "package exports/constants",
                }
            )
    test_groups: dict[str, list[str]] = {}
    for path in sorted((PROJECT_ROOT / "tests").rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        ]
        group = path.parent.relative_to(PROJECT_ROOT / "tests").as_posix() or "root"
        test_groups.setdefault(group, []).extend(names)
    for group, names in sorted(test_groups.items()):
        tests.append(
            {
                "suite": group,
                "test_count": len(names),
                "coverage_examples": ", ".join(name.removeprefix("test_") for name in names[:5]),
            }
        )
    return modules, tests


def _module_stage(relative: str) -> str:
    stages = (
        ("/models/", "Registry and model contracts"),
        ("/bundle/", "Bundle schemas and integrity"),
        ("/adapters/", "Model inference and classification metrics"),
        ("/onboarding/", "Read-only discovery and one-time export"),
        ("/domains/", "Domain feature compatibility and policy"),
        ("/scenarios/", "Controlled replay generation"),
        ("/monitoring/", "Deterministic data quality, drift, and performance"),
        ("/agent/", "Bounded agentic orchestration and governance"),
        ("/evaluation/", "Live-output evaluation"),
    )
    for token, stage in stages:
        if token in relative:
            return stage
    if relative.startswith("scripts/"):
        return "Command-line workflow"
    return "Project infrastructure"


def _live_evaluation_rows() -> list[dict[str, Any]]:
    frame = pd.read_csv(PROJECT_ROOT / "reports/evaluations/cross_model/scenario_comparison.csv")
    rows = []
    for record in frame.to_dict(orient="records"):
        rows.append(
            {
                "model_id": record["model_id"],
                "scenario": record["scenario_name"],
                "incident": record["incident_type"],
                "route": record["diagnostic_route"],
                "structured_output": bool(record["structured_output_success"]),
                "first_pass": bool(record["first_pass_verification"]),
                "revisions": int(record["revision_count"]),
                "fallback": bool(record["used_fallback"]),
                "approval_complete": bool(record["approval_completed"]),
                "latency_ms": float(record["latency_ms"]),
                "tokens": int(record["total_tokens"]),
            }
        )
    return rows


def reproduce_evidence() -> dict[str, Any]:
    generated_at = datetime.now(UTC).isoformat()
    registry = ModelRegistry()
    model_evidence: dict[str, Any] = {}
    feature_dictionary: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        bundle = RegisteredModelBundle(model_id, registry=registry)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            validation = validate_bundle(bundle)
        profile, metrics, dictionary = _profile_reference(bundle)
        feature_dictionary.extend(dictionary)
        model_evidence[model_id] = {
            "profile": profile,
            "metrics": metrics,
            "validation": {
                "valid": validation.valid,
                "checks": validation.checks,
                "errors": [item.model_dump(mode="json") for item in validation.errors],
                "warnings": [_safe_warning(item.message) for item in validation.warnings],
                "details": validation.details,
                "runtime_warnings": [_safe_warning(str(item.message)) for item in caught],
            },
            "specification": _model_specification(bundle),
        }
    scenario_results, scenario_design = _scenario_evidence(registry)
    credit_engine = MonitoringEngine(model_id="credit_default", registry=registry)
    malformed = credit_engine.bundle.load_reference_features().iloc[:1000].copy()
    malformed["LIMIT_BAL"] = malformed["LIMIT_BAL"].astype(object)
    malformed.loc[malformed.index[0], "LIMIT_BAL"] = "not-a-number"
    dtype_probe = evaluate_data_quality(
        malformed,
        credit_engine.schema,
        credit_engine.config["data_quality"],
    )
    modules, tests = _module_inventory()
    preservation_integrity = _preservation_integrity()
    test_execution = _run_test_suite()
    return {
        "generated_at_utc": generated_at,
        "git": {
            "head": _git_value("rev-parse", "HEAD"),
            "branch": _git_value("branch", "--show-current"),
            "tracked_worktree_status_before_generation": _git_value("status", "--short"),
        },
        "detected_project": {
            "project_type": (
                "Mixed operational analytics: binary-classifier monitoring, anomaly/drift "
                "detection, and governed agentic decision support"
            ),
            "primary_question": (
                "Can a new scored batch be trusted, has its distribution or labelled "
                "performance changed materially, and which bounded action should a human review?"
            ),
            "unit_of_analysis": (
                "One registered model plus one 1,000-row replay batch; model-level reference "
                "units are a credit-card account record or BRFSS respondent record."
            ),
            "analysis_moment": (
                "After a feature batch arrives and is scored; outcomes may be complete, partial, "
                "or absent."
            ),
            "claim_type": "Diagnostic and operational, with predictive inputs; not causal.",
            "primary_notebook": None,
        },
        "models": model_evidence,
        "feature_dictionary": feature_dictionary,
        "scenario_design": scenario_design,
        "scenario_results": scenario_results,
        "live_evaluation": _read_json("reports/evaluations/cross_model/cross_model_summary.json"),
        "live_evaluation_rows": _live_evaluation_rows(),
        "reliability_rerun": _read_json(
            "reports/evaluations/diabetes_risk/reliability_rerun/reliability_summary.json"
        ),
        "dtype_gate_probe": {
            "batch_valid": dtype_probe.batch_valid,
            "batch_blocked": dtype_probe.batch_blocked,
            "critical_evidence_ids": [
                item.evidence_id for item in dtype_probe.evidence if item.status == "critical"
            ],
            "interpretation": (
                "A critical incompatible dtype is recorded, but the current gate does not block "
                "inference. MonitoringEngine would proceed to the estimator and can fail."
            ),
        },
        "preservation_integrity": preservation_integrity,
        "test_execution": test_execution,
        "module_inventory": modules,
        "test_inventory": tests,
        "source_inputs": [
            "configs/model_registry.yaml",
            "configs/models/credit_default.yaml",
            "configs/models/diabetes_risk.yaml",
            "configs/monitoring.yaml",
            "configs/scenarios/credit_default.yaml",
            "configs/scenarios/diabetes_risk.yaml",
            "artifacts/baselines/reference_metrics.json",
            "registered_models/diabetes_risk/baselines/reference_metrics.json",
            "reports/evaluations/cross_model/cross_model_summary.json",
            "reports/evaluations/cross_model/scenario_comparison.csv",
            "reports/evaluations/diabetes_risk/reliability_rerun/reliability_summary.json",
            "reports/refactor/credit_preservation_manifest.json",
            "artifacts/metadata/bundle_manifest.json",
            "src/monitoring_agent/**/*.py",
            "scripts/*.py",
            "tests/**/*.py",
        ],
    }


def _source(source_id: str, label: str, path: str, description: str) -> dict[str, Any]:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        code = f"SELECT * FROM read_csv_auto('{path}', header=true)"
    elif suffix == ".json":
        code = f"SELECT * FROM read_json_auto('{path}')"
    else:
        code = f"SELECT content FROM read_text('{path}')"
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": code,
            "description": description,
            "tables_used": [path],
            "filters": ["Repository-local evidence only"],
        },
    }


def _format_parameters(values: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in values.items())


def _build_datasets(evidence: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    models = evidence["models"]
    live = evidence["live_evaluation"]
    preservation = evidence["preservation_integrity"]
    summary = [
        {
            "validated_models": sum(item["validation"]["valid"] for item in models.values()),
            "bundle_checks_passed": sum(
                sum(item["validation"]["checks"].values()) for item in models.values()
            ),
            "deterministic_scenarios": len(evidence["scenario_results"]),
            "live_scenarios_completed": live["completed_count"],
            "structured_output_rate": live["structured_output_success_rate"],
            "first_pass_rate": live["first_pass_verification_rate"],
            "fallback_rate": live["fallback_rate"],
            "known_material_defects": (
                int(not evidence["dtype_gate_probe"]["batch_blocked"])
                + len(preservation["mismatches"])
            ),
            "preservation_files_matched": preservation["matched_files"],
            "preservation_files_total": preservation["total_files"],
        }
    ]
    profiles = []
    model_specs = []
    threshold_rows: dict[str, list[dict[str, Any]]] = {}
    for model_id, item in models.items():
        profile = item["profile"]
        profiles.append(
            {
                "model_id": model_id,
                "unit": profile["unit"],
                "target": profile["target"],
                "rows": profile["rows"],
                "features": profile["raw_feature_count"],
                "transformed_features": profile["preprocessed_feature_count"],
                "positive_count": profile["positive_count"],
                "positive_rate": profile["positive_rate"],
                "duplicate_ids": profile["duplicate_record_ids"],
                "duplicate_rows": profile["exact_duplicate_rows"],
                "missing_cells": profile["missing_cells"],
                "infinite_cells": profile["infinite_numeric_cells"],
                "split": profile["split"],
            }
        )
        specification = item["specification"]
        model_specs.append(
            {
                "model_id": model_id,
                "artifact": specification["artifact_class"],
                "pipeline": " -> ".join(
                    f"{step['name']} ({step['class']})" for step in specification["pipeline_steps"]
                ),
                "classifier": specification["classifier_class"],
                "key_hyperparameters": _format_parameters(specification["classifier_parameters"]),
                "calibrator": specification["calibrator_class"] or "none",
                "fit_location": "Imported fitted artifact; no fitting in this repository.",
            }
        )
        rows = []
        for threshold_name, metrics in item["metrics"].items():
            label = "0.50 default" if threshold_name == "default_0_50" else "0.25 operating"
            for metric in ("precision", "recall", "specificity", "f1"):
                rows.append(
                    {
                        "threshold": label,
                        "metric": metric.replace("_", " ").title(),
                        "value": metrics[metric],
                        "sample_count": metrics["sample_count"],
                        "positive_count": metrics["positive_count"],
                        "model_id": model_id,
                    }
                )
        threshold_rows[model_id] = rows
    agent_rates = []
    for item in live["models"]:
        for label, field in (
            ("Structured output", "structured_output_success_rate"),
            ("First-pass verification", "first_pass_verification_rate"),
            ("Evidence grounding", "evidence_grounding_rate"),
            ("Policy compliance", "policy_compliance_rate"),
            ("Approval completion", "approval_completion_rate"),
            ("Fallback", "fallback_rate"),
        ):
            agent_rates.append(
                {
                    "model_id": item["model_id"],
                    "metric": label,
                    "rate": item[field],
                    "scenario_count": item["scenario_count"],
                }
            )
    metric_dictionary = [
        {
            "metric": "Accuracy",
            "applies_to": "Labelled classification",
            "definition": "Share of thresholded predictions equal to labels.",
            "direction": "Higher",
            "boundary": "Can hide minority-class failure; never called generic model accuracy here.",
        },
        {
            "metric": "Precision",
            "applies_to": "Labelled classification",
            "definition": "True positives divided by all predicted positives.",
            "direction": "Higher",
            "boundary": "Undefined when no positive prediction exists.",
        },
        {
            "metric": "Recall",
            "applies_to": "Labelled classification",
            "definition": "True positives divided by all observed positives.",
            "direction": "Higher",
            "boundary": "Threshold-dependent; source threshold 0.25 prioritizes recall.",
        },
        {
            "metric": "Specificity",
            "applies_to": "Explainer recomputation",
            "definition": "True negatives divided by all observed negatives.",
            "direction": "Higher",
            "boundary": "Not a stored monitoring policy metric; added for threshold interpretation.",
        },
        {
            "metric": "F1",
            "applies_to": "Labelled classification",
            "definition": "Harmonic mean of precision and recall.",
            "direction": "Higher",
            "boundary": "Does not encode domain-specific false-positive/false-negative costs.",
        },
        {
            "metric": "ROC-AUC",
            "applies_to": "Labelled classification",
            "definition": "Ranking discrimination across all thresholds.",
            "direction": "Higher",
            "boundary": "Can look optimistic under imbalance and is not generic accuracy.",
        },
        {
            "metric": "PR-AUC",
            "applies_to": "Labelled classification",
            "definition": "Average precision across recall levels.",
            "direction": "Higher",
            "boundary": "Baseline depends on positive prevalence; populations are not pooled.",
        },
        {
            "metric": "Brier score",
            "applies_to": "Probability quality",
            "definition": "Mean squared error of predicted probabilities.",
            "direction": "Lower",
            "boundary": "Captures calibration and discrimination jointly; no calibration curve/ECE is implemented.",
        },
        {
            "metric": "PSI",
            "applies_to": "Feature/prediction drift",
            "definition": "Reference-versus-current distribution change using reference bins.",
            "direction": "Lower",
            "boundary": "Thresholds are replay-MVP settings, not production-validated limits.",
        },
        {
            "metric": "Jensen-Shannon divergence",
            "applies_to": "Discrete drift",
            "definition": "Symmetric divergence between categorical distributions.",
            "direction": "Lower",
            "boundary": "Used with PSI and unseen-category rate.",
        },
        {
            "metric": "KS statistic",
            "applies_to": "Numeric/prediction drift",
            "definition": "Maximum empirical-CDF distance between samples.",
            "direction": "Lower",
            "boundary": "Supporting evidence only; p-values do not set materiality.",
        },
        {
            "metric": "Structured-output success",
            "applies_to": "Live agent evaluation",
            "definition": "Both live triage and recommendation calls parsed; fallback is excluded.",
            "direction": "Higher",
            "boundary": "One controlled run per scenario; not long-run reliability.",
        },
        {
            "metric": "First-pass verification",
            "applies_to": "Live agent evaluation",
            "definition": "Final deterministic verification passed with zero revision and no fallback.",
            "direction": "Higher",
            "boundary": "A revision can still yield a safe approved output.",
        },
        {
            "metric": "Evidence grounding",
            "applies_to": "Live agent evaluation",
            "definition": "All cited IDs exist and all claims/actions cite deterministic evidence.",
            "direction": "Higher",
            "boundary": "Does not score prose quality.",
        },
        {
            "metric": "Fallback rate",
            "applies_to": "Live agent evaluation",
            "definition": "Share of scenarios using the deterministic recommendation fallback.",
            "direction": "Lower",
            "boundary": "Fallback is a safety control, not a successful live structured output.",
        },
    ]
    future_scope = [
        {
            "priority": 1,
            "observed_gap": "The protected credit bundle manifest matches by size but not by SHA-256; only 21/22 preservation checks pass.",
            "proposed_work": "Recover or re-export the intended authoritative manifest through the governed source workflow, then update no hashes unless provenance proves the replacement.",
            "acceptance_evidence": "The protected file equals the approved source byte-for-byte and both preservation tests pass without weakening their assertions.",
        },
        {
            "priority": 2,
            "observed_gap": "Critical dtype/empty/infinite/integer findings can leave batch_blocked false.",
            "proposed_work": "Make every inference-safety critical finding block before estimator calls and add regression cases.",
            "acceptance_evidence": "Malformed batches return batch_blocked=true and MonitoringEngine never calls predict_scores.",
        },
        {
            "priority": 3,
            "observed_gap": "Both references are internal random held-out splits from public academic data.",
            "proposed_work": "Add later-period, external, or out-of-distribution evaluation with governed baselines.",
            "acceptance_evidence": "Versioned external/temporal benchmark with cohort definitions and uncertainty.",
        },
        {
            "priority": 4,
            "observed_gap": "Operating thresholds are imported; replay drift limits are placeholders.",
            "proposed_work": "Validate thresholds against domain false-positive/false-negative costs using validation-only selection.",
            "acceptance_evidence": "Approved cost matrix, validation selection record, locked test evaluation, and owner sign-off.",
        },
        {
            "priority": 5,
            "observed_gap": "Brier score is present, but reliability curves, ECE, and interval estimates are absent.",
            "proposed_work": "Add calibration plots, expected calibration error, and bootstrap uncertainty by model/subgroup.",
            "acceptance_evidence": "Calibration artifacts with sample sizes, intervals, and recalibration policy.",
        },
        {
            "priority": 6,
            "observed_gap": "Protected fields are declared but subgroup/fairness performance is not implemented.",
            "proposed_work": "Add governed subgroup metrics and stability tests only with suitable definitions and sample support.",
            "acceptance_evidence": "Approved groups, minimum counts, disparity measures, and review process.",
        },
        {
            "priority": 7,
            "observed_gap": "The authoritative live evaluation is one 12-case run; the repeat covers only two cases.",
            "proposed_work": "Run repeated provider trials with frozen prompts/evidence, failure taxonomy, and confidence intervals.",
            "acceptance_evidence": "Pre-registered trial count, isolated run labels, preserved failures, latency/token distributions.",
        },
        {
            "priority": 8,
            "observed_gap": "No production ingestion, identity/access control, case management, or observability integration exists.",
            "proposed_work": "Package a versioned service with immutable case IDs, authentication, audit logs, alerts, and SLOs.",
            "acceptance_evidence": "Load/security tests, rollback plan, monitored pilot, and human-acceptance criteria.",
        },
    ]
    return {
        "summary": summary,
        "reference_profiles": profiles,
        "model_specifications": model_specs,
        "credit_threshold_tradeoff": threshold_rows["credit_default"],
        "diabetes_threshold_tradeoff": threshold_rows["diabetes_risk"],
        "agent_rates": agent_rates,
        "scenario_design": evidence["scenario_design"],
        "scenario_results": evidence["scenario_results"],
        "feature_dictionary": evidence["feature_dictionary"],
        "live_evaluation_rows": evidence["live_evaluation_rows"],
        "module_inventory": evidence["module_inventory"],
        "test_inventory": evidence["test_inventory"],
        "preservation_mismatches": preservation["mismatches"],
        "metric_dictionary": metric_dictionary,
        "future_scope": future_scope,
    }


def _table(
    table_id: str,
    title: str,
    description: str,
    dataset: str,
    source_id: str,
    columns: list[dict[str, Any]],
    sort_field: str,
    direction: str = "asc",
) -> dict[str, Any]:
    return {
        "id": table_id,
        "title": title,
        "description": description,
        "dataset": dataset,
        "sourceId": source_id,
        "defaultSort": {"field": sort_field, "direction": direction},
        "columns": columns,
    }


def build_artifact(evidence: dict[str, Any]) -> dict[str, Any]:
    datasets = _build_datasets(evidence)
    generated_at = evidence["generated_at_utc"]
    sources = [
        _source(
            "current_reproduction",
            "Current explainer reproduction snapshot",
            "reports/project_explainer_evidence.json",
            "Current bundle validation, reference profiling, independently recomputed metrics, deterministic scenario reruns, and code/test inventories.",
        ),
        _source(
            "credit_metrics",
            "Credit-default reference metrics",
            "artifacts/baselines/reference_metrics.json",
            "Stored held-out credit-default metrics and threshold results.",
        ),
        _source(
            "diabetes_metrics",
            "Diabetes-risk reference metrics",
            "registered_models/diabetes_risk/baselines/reference_metrics.json",
            "Stored held-out BRFSS diabetes-risk metrics and threshold results.",
        ),
        _source(
            "cross_model_eval",
            "Preserved cross-model live evaluation",
            "reports/evaluations/cross_model/cross_model_summary.json",
            "Authoritative first-run evaluation across two models and six scenarios per model.",
        ),
        _source(
            "cross_model_rows",
            "Cross-model scenario comparison",
            "reports/evaluations/cross_model/scenario_comparison.csv",
            "Per-model, per-scenario live-agent outcomes used in the aggregate evaluation.",
        ),
        _source(
            "reliability_rerun",
            "Isolated diabetes reliability repeat",
            "reports/evaluations/diabetes_risk/reliability_rerun/reliability_summary.json",
            "Two-case repeat evidence that preserves rather than replaces the original evaluation.",
        ),
        _source(
            "registry_contract",
            "Registered-model contracts",
            "configs/model_registry.yaml",
            "Registry entry point for the two strict binary-classification manifests.",
        ),
        _source(
            "monitoring_code",
            "Deterministic monitoring engine",
            "src/monitoring_agent/monitoring/engine.py",
            "Authoritative orchestration for data-quality gating, scoring, drift, performance, evidence, and incident candidates.",
        ),
        _source(
            "agent_graph",
            "Bounded LangGraph workflow",
            "src/monitoring_agent/agent/graph.py",
            "Authoritative graph topology for triage, diagnostics, recommendation, verification, fallback, and approval.",
        ),
        _source(
            "agent_policy",
            "Deterministic agent policy and verifier",
            "src/monitoring_agent/agent/policy.py",
            "Hard policy checks governing evidence, labels, actions, revisions, fallback, and human approval.",
        ),
        _source(
            "scenario_configs",
            "Controlled replay definitions",
            "configs/scenarios/credit_default.yaml",
            "Model-aware replay definitions paired with configs/scenarios/diabetes_risk.yaml.",
        ),
    ]
    cards = [
        {
            "id": "validated_models",
            "description": "Both registered inference bundles passed current generic contract, alignment, threshold, and probability-reproduction validation; the separate preservation gate is reported below.",
            "dataset": "summary",
            "sourceId": "current_reproduction",
            "metrics": [
                {"label": "Validated bundles", "field": "validated_models", "format": "number"}
            ],
        },
        {
            "id": "bundle_checks",
            "description": "Fifteen generic checks passed for each bundle in the current environment.",
            "dataset": "summary",
            "sourceId": "current_reproduction",
            "metrics": [
                {
                    "label": "Bundle checks passed",
                    "field": "bundle_checks_passed",
                    "format": "number",
                }
            ],
        },
        {
            "id": "deterministic_scenarios",
            "description": "Six current replay scenarios were reproduced for each model without rewriting historical reports.",
            "dataset": "summary",
            "sourceId": "current_reproduction",
            "metrics": [
                {
                    "label": "Scenario outcomes reproduced",
                    "field": "deterministic_scenarios",
                    "format": "number",
                }
            ],
        },
        {
            "id": "live_completed",
            "description": "All scenarios reached a completed state in the preserved first-run live evaluation.",
            "dataset": "summary",
            "sourceId": "cross_model_eval",
            "metrics": [
                {
                    "label": "Live scenarios completed",
                    "field": "live_scenarios_completed",
                    "format": "number",
                }
            ],
        },
        {
            "id": "structured_output",
            "description": "Both provider calls parsed successfully; deterministic fallback is deliberately excluded.",
            "dataset": "summary",
            "sourceId": "cross_model_eval",
            "metrics": [
                {
                    "label": "Structured-output success",
                    "field": "structured_output_rate",
                    "format": "percent",
                }
            ],
        },
        {
            "id": "first_pass",
            "description": "Final verification passed without revision or fallback in nine of twelve live scenarios.",
            "dataset": "summary",
            "sourceId": "cross_model_eval",
            "metrics": [
                {
                    "label": "First-pass verification",
                    "field": "first_pass_rate",
                    "format": "percent",
                }
            ],
        },
        {
            "id": "fallback",
            "description": "Two diabetes scenarios used the conservative deterministic fallback after live-provider/output failures.",
            "dataset": "summary",
            "sourceId": "cross_model_eval",
            "metrics": [{"label": "Fallback rate", "field": "fallback_rate", "format": "percent"}],
        },
        {
            "id": "known_material_defects",
            "description": "One protected-file hash mismatch and one unsafe inference-gate behavior remain unresolved.",
            "dataset": "summary",
            "sourceId": "current_reproduction",
            "metrics": [
                {
                    "label": "Known material defects",
                    "field": "known_material_defects",
                    "format": "number",
                }
            ],
        },
    ]
    charts = [
        {
            "id": "credit_threshold_chart",
            "title": "Credit-default threshold metrics",
            "subtitle": "Same 6,002-row held-out population; values are rates from 0 to 1.",
            "type": "bar",
            "dataset": "credit_threshold_tradeoff",
            "sourceId": "credit_metrics",
            "encodings": {
                "x": {"field": "metric", "type": "ordinal", "label": "Metric"},
                "y": {
                    "field": "value",
                    "type": "quantitative",
                    "label": "Rate",
                    "format": "percent",
                },
                "color": {"field": "threshold", "type": "nominal", "label": "Threshold"},
            },
            "yAxisTitle": "Rate",
            "valueFormat": "percent",
            "layout": "full",
        },
        {
            "id": "diabetes_threshold_chart",
            "title": "Diabetes-risk threshold metrics",
            "subtitle": "Same 50,736-row held-out population; values are rates from 0 to 1.",
            "type": "bar",
            "dataset": "diabetes_threshold_tradeoff",
            "sourceId": "diabetes_metrics",
            "encodings": {
                "x": {"field": "metric", "type": "ordinal", "label": "Metric"},
                "y": {
                    "field": "value",
                    "type": "quantitative",
                    "label": "Rate",
                    "format": "percent",
                },
                "color": {"field": "threshold", "type": "nominal", "label": "Threshold"},
            },
            "yAxisTitle": "Rate",
            "valueFormat": "percent",
            "layout": "full",
        },
        {
            "id": "agent_rate_chart",
            "title": "Live-agent evaluation rates by registered model",
            "subtitle": "Six controlled scenarios per model; fallback is a separate safety outcome.",
            "type": "bar",
            "dataset": "agent_rates",
            "sourceId": "cross_model_eval",
            "encodings": {
                "x": {"field": "metric", "type": "ordinal", "label": "Evaluation metric"},
                "y": {
                    "field": "rate",
                    "type": "quantitative",
                    "label": "Scenario rate",
                    "format": "percent",
                },
                "color": {"field": "model_id", "type": "nominal", "label": "Model"},
            },
            "yAxisTitle": "Scenario rate",
            "valueFormat": "percent",
            "layout": "full",
        },
    ]
    tables = [
        _table(
            "reference_profiles_table",
            "Reference data quality and grain",
            "Current local profiles for each held-out reference bundle.",
            "reference_profiles",
            "current_reproduction",
            [
                {"field": "model_id", "label": "Model", "type": "text"},
                {"field": "unit", "label": "Unit", "type": "text"},
                {"field": "target", "label": "Target", "type": "text"},
                {"field": "rows", "label": "Rows", "format": "number"},
                {"field": "features", "label": "Raw features", "format": "number"},
                {"field": "transformed_features", "label": "Transformed", "format": "number"},
                {"field": "positive_count", "label": "Positive labels", "format": "number"},
                {"field": "positive_rate", "label": "Positive rate", "format": "percent"},
                {"field": "duplicate_ids", "label": "Duplicate IDs", "format": "number"},
                {"field": "missing_cells", "label": "Missing cells", "format": "number"},
                {"field": "infinite_cells", "label": "Infinite cells", "format": "number"},
                {"field": "split", "label": "Evaluation split", "type": "text"},
            ],
            "model_id",
        ),
        _table(
            "model_specs_table",
            "Imported fitted-model specifications",
            "Hyperparameters are introspected from the fitted local artifacts; no fitting occurs here.",
            "model_specifications",
            "current_reproduction",
            [
                {"field": "model_id", "label": "Model", "type": "text"},
                {"field": "pipeline", "label": "Pipeline", "type": "text"},
                {"field": "classifier", "label": "Classifier", "type": "text"},
                {
                    "field": "key_hyperparameters",
                    "label": "Key fitted hyperparameters",
                    "type": "text",
                },
                {"field": "calibrator", "label": "Calibrator", "type": "text"},
                {"field": "fit_location", "label": "Fit boundary", "type": "text"},
            ],
            "model_id",
        ),
        _table(
            "metric_dictionary_table",
            "Metric dictionary and evidence boundary",
            "Classification, drift, probability, and live-agent metrics actually used by this project.",
            "metric_dictionary",
            "current_reproduction",
            [
                {"field": "metric", "label": "Metric", "type": "text"},
                {"field": "applies_to", "label": "Applies to", "type": "text"},
                {"field": "definition", "label": "Definition", "type": "text"},
                {"field": "direction", "label": "Preferred direction", "type": "text"},
                {"field": "boundary", "label": "Important limitation", "type": "text"},
            ],
            "metric",
        ),
        _table(
            "feature_dictionary_table",
            "Complete registered feature dictionary",
            "All 57 raw inference fields across both registered model contracts.",
            "feature_dictionary",
            "current_reproduction",
            [
                {"field": "model_id", "label": "Model", "type": "text"},
                {"field": "position", "label": "Position", "format": "number"},
                {"field": "field", "label": "Field", "type": "text"},
                {"field": "kind", "label": "Kind", "type": "text"},
                {"field": "dtype", "label": "dtype", "type": "text"},
                {"field": "integer_constrained", "label": "Integer", "type": "boolean"},
                {"field": "nullable", "label": "Nullable", "type": "boolean"},
                {"field": "observed_range", "label": "Observed reference range", "type": "text"},
                {"field": "allowed_values", "label": "Allowed values", "type": "text"},
                {"field": "meaning", "label": "Meaning", "type": "text"},
            ],
            "model_id",
        ),
        _table(
            "scenario_design_table",
            "Controlled scenario design",
            "Two model-specific definitions for each of six replay scenarios.",
            "scenario_design",
            "scenario_configs",
            [
                {"field": "model_id", "label": "Model", "type": "text"},
                {"field": "scenario", "label": "Scenario", "type": "text"},
                {"field": "simulation_type", "label": "Simulation", "type": "text"},
                {"field": "sample_size", "label": "Rows", "format": "number"},
                {"field": "seed", "label": "Seed", "format": "number"},
                {"field": "labels", "label": "Labels", "type": "text"},
                {"field": "configured_change", "label": "Configured change", "type": "text"},
                {"field": "expected_candidates", "label": "Expected incident", "type": "text"},
            ],
            "model_id",
        ),
        _table(
            "scenario_results_table",
            "Current deterministic replay results",
            "Recomputed in memory; historical generated reports were not overwritten.",
            "scenario_results",
            "current_reproduction",
            [
                {"field": "model_id", "label": "Model", "type": "text"},
                {"field": "scenario", "label": "Scenario", "type": "text"},
                {"field": "batch_blocked", "label": "Blocked", "type": "boolean"},
                {
                    "field": "performance_evaluated",
                    "label": "Performance evaluated",
                    "type": "boolean",
                },
                {"field": "label_coverage", "label": "Label coverage", "format": "percent"},
                {
                    "field": "critical_features",
                    "label": "Critical drift features",
                    "format": "number",
                },
                {"field": "incident_candidates", "label": "Incident candidates", "type": "text"},
                {"field": "overall_severity", "label": "Severity", "type": "text"},
                {"field": "operating_recall", "label": "Recall @0.25", "format": "percent"},
                {"field": "operating_precision", "label": "Precision @0.25", "format": "percent"},
                {"field": "operating_f1", "label": "F1 @0.25", "format": "percent"},
                {"field": "pr_auc", "label": "PR-AUC", "format": "number"},
                {"field": "brier_score", "label": "Brier", "format": "number"},
            ],
            "model_id",
        ),
        _table(
            "live_evaluation_table",
            "Preserved live-agent scenario outcomes",
            "One authoritative first-run result for each model and scenario.",
            "live_evaluation_rows",
            "cross_model_rows",
            [
                {"field": "model_id", "label": "Model", "type": "text"},
                {"field": "scenario", "label": "Scenario", "type": "text"},
                {"field": "incident", "label": "Incident", "type": "text"},
                {"field": "route", "label": "Route", "type": "text"},
                {"field": "structured_output", "label": "Structured", "type": "boolean"},
                {"field": "first_pass", "label": "First pass", "type": "boolean"},
                {"field": "revisions", "label": "Revisions", "format": "number"},
                {"field": "fallback", "label": "Fallback", "type": "boolean"},
                {"field": "approval_complete", "label": "Approval", "type": "boolean"},
                {"field": "latency_ms", "label": "LLM latency (ms)", "format": "number"},
                {"field": "tokens", "label": "Tokens", "format": "number"},
            ],
            "model_id",
        ),
        _table(
            "module_inventory_table",
            "Module-by-module implementation map",
            "No notebook exists; this is the equivalent executable walkthrough.",
            "module_inventory",
            "current_reproduction",
            [
                {"field": "stage", "label": "Workflow stage", "type": "text"},
                {"field": "path", "label": "Module", "type": "text"},
                {"field": "purpose", "label": "Purpose", "type": "text"},
                {"field": "key_objects", "label": "Key public objects", "type": "text"},
            ],
            "stage",
        ),
        _table(
            "test_inventory_table",
            "Test-suite map",
            "Current static inventory of test functions by suite.",
            "test_inventory",
            "current_reproduction",
            [
                {"field": "suite", "label": "Suite", "type": "text"},
                {"field": "test_count", "label": "Tests", "format": "number"},
                {"field": "coverage_examples", "label": "Coverage examples", "type": "text"},
            ],
            "suite",
        ),
        _table(
            "preservation_mismatches_table",
            "Protected evidence mismatches",
            "Current byte-level comparison against the credit preservation manifest; protected files are never rewritten here.",
            "preservation_mismatches",
            "current_reproduction",
            [
                {"field": "relative_path", "label": "Protected file", "type": "text"},
                {"field": "exists", "label": "Exists", "type": "boolean"},
                {"field": "expected_size", "label": "Expected bytes", "format": "number"},
                {"field": "actual_size", "label": "Actual bytes", "format": "number"},
                {"field": "size_matches", "label": "Size matches", "type": "boolean"},
                {"field": "expected_sha256", "label": "Expected SHA-256", "type": "text"},
                {"field": "actual_sha256", "label": "Actual SHA-256", "type": "text"},
                {"field": "sha256_matches", "label": "SHA matches", "type": "boolean"},
            ],
            "relative_path",
        ),
        _table(
            "future_scope_table",
            "Project-specific future scope",
            "Prioritized from current observed gaps; no future metric is claimed.",
            "future_scope",
            "current_reproduction",
            [
                {"field": "priority", "label": "Priority", "format": "number"},
                {"field": "observed_gap", "label": "Observed gap", "type": "text"},
                {"field": "proposed_work", "label": "Proposed work", "type": "text"},
                {
                    "field": "acceptance_evidence",
                    "label": "Evidence needed to call it complete",
                    "type": "text",
                },
            ],
            "priority",
        ),
    ]
    test_run = evidence["test_execution"]
    test_counts = test_run["counts"]
    collected_cases = sum(test_counts[name] for name in ("passed", "failed", "skipped"))
    failed_nodes = ", ".join(f"`{name}`" for name in test_run["failed_nodes"]) or "none"
    static_test_functions = sum(item["test_count"] for item in evidence["test_inventory"])
    blocks = [
        {"id": "title", "type": "markdown", "body": "# Agentic Model Risk & Monitoring Copilot"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": "current_reproduction",
            "body": "## Technical summary\n\n**This repository monitors two already-fitted binary classifiers; it does not train a model.** A strict registry loads self-contained credit-default and BRFSS diabetes-risk bundles, validates feature order and probability reproduction, replays six controlled 1,000-row scenarios per model, calculates deterministic data-quality/drift/performance evidence, and passes a bounded evidence packet to one LangGraph orchestrator for recommendation, verification, fallback, and human approval.\n\nCurrent reproduction validates **2/2 bundles**, passes **30/30 generic bundle checks**, and recreates **12/12 deterministic scenario outcomes**. The preserved first live-provider evaluation completed **12/12** scenarios with **83.33% structured-output success**, **75% first-pass verification**, and **16.66% deterministic fallback**. These are controlled replay results, not production reliability or model-quality estimates.\n\n**Two material defects remain.** First, the protected credit `artifacts/metadata/bundle_manifest.json` matches its recorded 6,019-byte size but not its required SHA-256, so only **21/22 preservation checks** pass. Second, a malformed credit feature produces critical `DQ-DTYPE-LIMIT-BAL` evidence while `batch_blocked` remains false, so inference can still be attempted. The explainer therefore assesses the project as technically coherent but **not production-ready**.",
        },
        {
            "id": "headline_metrics",
            "type": "metric-strip",
            "cardIds": [card["id"] for card in cards],
        },
        {
            "id": "detected_task",
            "type": "markdown",
            "sourceId": "current_reproduction",
            "body": "## This is diagnostic monitoring plus governed decision support\n\n- **Primary analytical task:** detect data-quality failures, covariate/prediction drift, labelled performance degradation, and insufficient evidence around existing binary classifiers.\n- **Business question:** can the current batch be trusted, what changed relative to a registered held-out baseline, and which non-executing action should a human review?\n- **Unit of analysis:** one registered model and one batch; individual reference units are a credit-card account record or a BRFSS respondent screening record.\n- **Decision moment:** after a feature batch arrives and is scored; labels may be complete, partial, or absent.\n- **Claim type:** descriptive and diagnostic evidence feeds operational recommendations. The fitted classifiers are predictive; the monitoring design is not causal.\n\nRegression, forecasting, clustering, and optimization diagnostics are intentionally omitted. No R², Adjusted R², residual tests, stationarity tests, or random model-training claims belong in this project.",
        },
        {
            "id": "architecture",
            "type": "markdown",
            "sourceId": "monitoring_code",
            "body": "## Deterministic evidence remains authoritative throughout the workflow\n\n```text\nstrict registry + bundle manifest\n        |\nreference features, labels, predictions + current batch\n        |\ndata quality gate -> fitted classifier score (if not blocked)\n        |\nfeature/prediction drift + labelled performance (when eligible)\n        |\nstable evidence IDs + deterministic incident candidates\n        |\nbounded evidence packet (maximum 30 records)\n        |\nLLM triage -> controlled diagnostic route -> LLM recommendation\n        |\ndeterministic citation/policy verifier\n     revise once | fallback | pass\n        |\nnon-normal cases pause for explicit human approval\n```\n\nThe LLM never receives raw feature rows, recalculates metrics, executes remediation, retrains models, changes thresholds, or bypasses the deterministic verifier.",
        },
        {
            "id": "data_scope",
            "type": "markdown",
            "sourceId": "current_reproduction",
            "body": "## Two clean held-out references support replay, not deployment claims\n\nThe credit bundle contains **6,002 rows**, **36 ordered inference features**, and **1,327 positives (22.11%)** from a group-aware stratified random held-out split. The diabetes bundle contains **50,736 rows**, **21 raw predictors**, **52 post-engineering features**, and **7,069 positives (13.93%)** from a stratified random held-out split. Current profiling finds zero duplicate record IDs, zero exact duplicate rows, zero missing cells, and zero infinite numeric cells in both references.\n\nThose checks establish internal bundle cleanliness and alignment. They do not establish temporal representativeness, external validity, fairness, production freshness, or prospective performance.",
        },
        {
            "id": "reference_profiles",
            "type": "table",
            "tableId": "reference_profiles_table",
            "layout": "full",
        },
        {
            "id": "model_boundary",
            "type": "markdown",
            "sourceId": "current_reproduction",
            "body": "## The models are imported fitted artifacts with explicit preprocessing boundaries\n\nThe credit artifact is a scikit-learn `Pipeline` with a `ColumnTransformer` and XGBoost classifier. The diabetes artifact wraps a fitted feature-engineering pipeline and XGBoost classifier with a fitted sigmoid calibrator. The current repository performs **no fitting, feature selection, hyperparameter search, calibration fitting, or threshold selection**; it reproduces source predictions and retains source-authoritative thresholds.\n\nThis boundary prevents the monitoring layer from silently changing model behavior. It also means training provenance, tuning validity, explainability, and source-project leakage controls must be evaluated in the originating model repositories rather than inferred here.",
        },
        {"id": "model_specs", "type": "table", "tableId": "model_specs_table", "layout": "full"},
        {
            "id": "credit_threshold_explanation",
            "type": "markdown",
            "sourceId": "credit_metrics",
            "body": "## Credit threshold 0.25 raises recall at a material precision cost\n\nOn the same 6,002-row held-out credit population, moving from the default **0.50** threshold to the source operating threshold **0.25** increases recall from **34.14% to 58.10%**, while precision falls from **65.84% to 47.77%** and specificity from **94.97% to 81.97%**. F1 rises from **44.96% to 52.43%**.\n\nThis is a threshold trade-off, not a new model fit. ROC-AUC (**0.7748**), PR-AUC (**0.5415**), and Brier score (**0.1371**) are threshold-independent for these stored probabilities. Whether 0.25 is operationally preferable depends on false-negative and false-positive costs that this repository does not validate.",
        },
        {
            "id": "credit_threshold_chart_block",
            "type": "chart",
            "chartId": "credit_threshold_chart",
            "layout": "full",
        },
        {
            "id": "diabetes_threshold_explanation",
            "type": "markdown",
            "sourceId": "diabetes_metrics",
            "body": "## Diabetes threshold 0.25 prioritizes screening recall over precision\n\nOn the same 50,736-row BRFSS held-out population, moving from **0.50** to **0.25** raises recall from **14.00% to 58.99%**, while precision moves from **58.17% to 38.65%** and specificity from **98.37% to 84.84%**. F1 rises from **22.57% to 46.70%**.\n\nROC-AUC is **0.8286**, PR-AUC is **0.4294**, and Brier score is **0.0970**. These are internal survey-model results, not diagnosis, treatment validity, external clinical validation, or patient-level benefit.",
        },
        {
            "id": "diabetes_threshold_chart_block",
            "type": "chart",
            "chartId": "diabetes_threshold_chart",
            "layout": "full",
        },
        {
            "id": "metric_methods",
            "type": "markdown",
            "sourceId": "monitoring_code",
            "body": "## Metrics are selected for classification monitoring and evidence sufficiency\n\nPerformance is evaluated only when labels have the exact `[record_id, actual_label]` schema, align to feature IDs, contain both binary classes, and meet the **200-label** policy minimum. Threshold metrics include accuracy, precision, recall, F1, positive counts/rates, and the confusion matrix. ROC-AUC and PR-AUC measure ranking; Brier score measures probability error. Specificity is independently recomputed in this explainer to make the threshold cost visible.\n\nDrift uses PSI as the main materiality signal, Jensen-Shannon divergence and unseen-category rate for discrete fields, and KS as supporting evidence. A small KS p-value is not treated as material drift by itself. Missing labels or only 100/1,000 labels produce `insufficient_evidence`; the system does not fabricate performance metrics.",
        },
        {
            "id": "metric_dictionary",
            "type": "table",
            "tableId": "metric_dictionary_table",
            "layout": "full",
        },
        {
            "id": "feature_dictionary_intro",
            "type": "markdown",
            "sourceId": "registry_contract",
            "body": "## The feature contracts are explicit, ordered, and target-free\n\nEvery inference input must contain the identifier followed by the exact registered feature order. The target is stored separately; the adapter rejects target leakage, missing/extra columns, and order mismatches. Observed minima and maxima are held-out monitoring baselines, not universal domain-validity rules.\n\nThe table below lists all raw fields. Several credit fields are engineered utilization, payment, and delay summaries. The diabetes pipeline deterministically expands 21 raw BRFSS indicators into 52 features before applying its fitted classifier.",
        },
        {
            "id": "feature_dictionary",
            "type": "table",
            "tableId": "feature_dictionary_table",
            "layout": "full",
        },
        {
            "id": "scenario_design_intro",
            "type": "markdown",
            "sourceId": "scenario_configs",
            "body": "## Six scenarios isolate stable, drift, corruption, outcome, and label-availability behavior\n\nEach model uses a fixed seed (`20260725`) and a stratified 1,000-row replay sample. Feature-drift scenarios change model-specific valid-range features; the corruption scenario injects duplicate IDs, missingness, and observed-range violations; performance degradation flips low-score negative labels; unlabelled drift removes outcomes; insufficient labels retains only 100 aligned outcomes.\n\nThese transformations are controlled tests of workflow behavior. They do not estimate incident prevalence, causal mechanisms, or future production performance.",
        },
        {
            "id": "scenario_design",
            "type": "table",
            "tableId": "scenario_design_table",
            "layout": "full",
        },
        {
            "id": "deterministic_results_intro",
            "type": "markdown",
            "sourceId": "current_reproduction",
            "body": "## Current deterministic reruns recover the intended evidence boundaries\n\nBoth normal-operation replays remain noncritical. Both corruption scenarios block inference and suppress drift/performance evaluation. Both labelled feature-drift scenarios combine critical drift with performance effects. Both synthetic outcome-shift scenarios produce performance degradation without feature drift. Both unlabelled cases retain drift evidence but abstain from performance claims, and both 100-label cases route to insufficient evidence.\n\nScenario recall, precision, F1, PR-AUC, and Brier values are scenario-specific diagnostics. They must not replace full-reference model metrics or be averaged across the two domains.",
        },
        {
            "id": "deterministic_results",
            "type": "table",
            "tableId": "scenario_results_table",
            "layout": "full",
        },
        {
            "id": "agent_workflow",
            "type": "markdown",
            "sourceId": "agent_graph",
            "body": "## One bounded agent adds synthesis without owning the facts\n\nThe graph validates the monitoring result, selects all critical/warning/system evidence within a 30-record cap, requests strict structured triage, filters already-computed evidence by a controlled route, requests a structured recommendation, and then verifies every citation, action, severity, uncertainty, and governance rule. At most one revision is allowed. Provider failure, parsing failure, or a still-invalid recommendation produces a conservative deterministic fallback.\n\nNormal operation can finish without approval. Every non-normal recommendation and each proposed action requires an interrupt/resume approval decision. SQLite checkpointing supports local durable demonstrations but is not a production case store.",
        },
        {
            "id": "agent_rate_explanation",
            "type": "markdown",
            "sourceId": "cross_model_eval",
            "body": "## The first live evaluation validates controls but exposes provider sensitivity\n\nAcross six scenarios per model, incident compatibility, route compatibility, evidence grounding, policy compliance, and approval completion are all **100%**. Credit produced **100%** structured output and **0%** fallback; diabetes produced **66.67%** structured output and **33.33%** fallback. Combined first-pass verification is **75%**.\n\nThese rates describe exactly twelve controlled calls. They do not estimate rare provider failures, repeat variance, production availability, prose quality, latency SLOs, or operational usefulness.",
        },
        {
            "id": "agent_rate_chart_block",
            "type": "chart",
            "chartId": "agent_rate_chart",
            "layout": "full",
        },
        {
            "id": "live_evaluation",
            "type": "table",
            "tableId": "live_evaluation_table",
            "layout": "full",
        },
        {
            "id": "failure_analysis",
            "type": "markdown",
            "sourceId": "reliability_rerun",
            "body": "## Failure cases remain evidence, even when a repeat succeeds\n\nThe original diabetes feature-drift recommendation hit Groq HTTP 429 and used fallback. The original unlabelled-drift run first rejected an incompatible triage and later hit HTTP 429, also using fallback. A separately labelled two-case repeat did not reproduce either provider error: feature drift passed first try, while unlabelled drift needed one bounded revision.\n\nThe repeat achieved 100% structured output and 0% fallback for only two cases, with 50% first-pass verification. It does **not** replace the original 12-case evaluation, erase the failures, or establish production reliability. Mean repeat LLM latency was 43.82 seconds, which also underscores the absence of an approved latency budget.",
        },
        {
            "id": "module_walkthrough_intro",
            "type": "markdown",
            "sourceId": "current_reproduction",
            "body": "## No notebook exists; the executable walkthrough is module-by-module\n\nThe empty `notebooks/` directory contains only `.gitkeep`. The table therefore replaces cell-by-cell commentary with every Python module and script, its workflow stage, module docstring, and public objects. The intended reading path is registry/bundles -> adapters/onboarding -> scenarios -> deterministic monitoring -> evidence packet -> graph/policy/verifier -> evaluation/reporting -> CLI scripts.",
        },
        {
            "id": "module_inventory",
            "type": "table",
            "tableId": "module_inventory_table",
            "layout": "full",
        },
        {
            "id": "test_walkthrough",
            "type": "markdown",
            "sourceId": "current_reproduction",
            "body": f"## Tests focus on contracts, evidence, policy, and safe interruption\n\nThe suite covers registry integrity, exact feature order, target leakage, probability reproduction, scenario determinism, partial labels, data quality, drift, performance, evidence selection, graph routing, hard policy, revision/fallback, checkpoint resume, onboarding, and live/cross-model evaluation. The static inventory below contains **{static_test_functions} test functions**, which expand through parametrization into **{collected_cases} collected cases** in the current full run. That run produced **{test_counts['passed']} passes and {test_counts['failed']} failures**. Failed nodes: {failed_nodes}. In this snapshot both failures trace to the single protected credit bundle-manifest SHA mismatch shown below.\n\nA separate important gap is explicit: existing data-quality tests do not cover malformed dtypes, non-finite values, completely empty features, or non-integer constrained values all the way through the inference gate.",
        },
        {
            "id": "test_inventory",
            "type": "table",
            "tableId": "test_inventory_table",
            "layout": "full",
        },
        {
            "id": "preservation_mismatches_intro",
            "type": "markdown",
            "sourceId": "current_reproduction",
            "body": "## Preservation is a failing gate, not a warning to normalize away\n\nThe credit preservation manifest declares 22 files byte-identical. Current direct hashing matches 21. `artifacts/metadata/bundle_manifest.json` still has the expected 6,019 bytes but its actual SHA-256 is `5031263b0d38ba8967b9c883c70d7aca54408ca59da72015d310dd48b1105234`, not the protected `5a8feb5eaff711a86fed179105f0fcd93f507f7faf12abb9fa1f38d1fa42e677`. The file is gitignored and no matching copy was found elsewhere in the local project tree; its intended bytes cannot be reconstructed safely from size alone. The explainer records the mismatch and leaves the evidence untouched.",
        },
        {
            "id": "preservation_mismatches",
            "type": "table",
            "tableId": "preservation_mismatches_table",
            "layout": "full",
        },
        {
            "id": "known_limitations",
            "type": "markdown",
            "sourceId": "current_reproduction",
            "body": "## The strongest conclusion is controlled-replay validity with material caveats\n\n**Current validation supports:** two loadable bundles, exact local probability reproduction, deterministic scenario behavior, evidence-ID grounding, bounded structured outputs, one revision, deterministic fallback, and explicit approval controls.\n\n**It does not support:** protected-artifact integrity for all credit files; production readiness; live traffic monitoring; causal diagnosis; automated remediation; model retraining in this repository; external/temporal validation; fairness claims; robust calibration surveillance; authenticated case management; repeated-provider reliability; production latency/cost SLOs; or legal, clinical, or credit-decision use.\n\n**Known defects:** the protected credit bundle-manifest hash fails 2 test assertions, and critical dtype/empty/infinite/integer findings do not consistently set `batch_blocked=true`. The current malformed `LIMIT_BAL` probe returns `batch_valid=false`, `batch_blocked=false`, and `DQ-DTYPE-LIMIT-BAL`; an estimator call can then fail. These are reproducibility and operational-safety gaps, not cosmetic limitations.",
        },
        {
            "id": "future_scope_intro",
            "type": "markdown",
            "sourceId": "current_reproduction",
            "body": "## Future work follows directly from observed gaps\n\nPriority begins with governed recovery of the protected manifest and then the inference gate, which can convert an already-detected critical defect into an estimator crash. External validation, governed threshold costs, calibration/fairness monitoring, repeated-provider trials, and production controls follow. Every item below is proposed work; no future metric or readiness state is invented.",
        },
        {"id": "future_scope", "type": "table", "tableId": "future_scope_table", "layout": "full"},
        {
            "id": "reproducibility",
            "type": "markdown",
            "sourceId": "current_reproduction",
            "body": "## Reproduce the evidence without touching historical evaluations\n\nFrom the repository root on Windows:\n\n```powershell\n.\\.venv\\Scripts\\python.exe scripts\\verify_setup.py\n.\\.venv\\Scripts\\python.exe scripts\\validate_credit_default_bundle.py\n.\\.venv\\Scripts\\python.exe -m pytest\n.\\.venv\\Scripts\\python.exe scripts\\build_project_explainer.py\n```\n\nThe explainer generator validates both bundles, profiles reference Parquet files, independently recomputes threshold metrics, reruns all 12 deterministic scenarios in memory, reproduces the malformed-dtype probe, inventories modules/tests, writes `reports/project_explainer_evidence.json`, writes `project_explainer_artifact.json`, and packages `project_explainer.html`. It never calls Groq, reads secrets, retrains, regenerates scenarios, or overwrites historical evaluation evidence. Use `--archive-existing` before a future intentional replacement.",
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": "## Questions that must be answered before operational use\n\n1. Who owns each model, drift threshold, incident decision, and approval SLA?\n2. What are the domain-specific costs of false negatives and false positives at the decision moment?\n3. Which prospective or external population should become the governed baseline, and how often may it change?\n4. Which protected-group definitions and minimum sample sizes are ethically and legally appropriate?\n5. What reliability, latency, token-cost, retention, security, and audit requirements define an acceptable pilot?",
        },
    ]
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Agentic Model Risk & Monitoring Copilot",
            "description": "Technical, evidence-backed explainer of the complete monitoring repository.",
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": [
                {"id": item["id"], "label": item["label"], "path": item["path"]} for item in sources
            ],
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": datasets,
        },
        "sources": sources,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _archive_existing() -> Path | None:
    existing = [path for path in (ARTIFACT_PATH, EVIDENCE_PATH, HTML_PATH) if path.is_file()]
    if not existing:
        return None
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = ARCHIVE_ROOT / stamp
    destination.mkdir(parents=True, exist_ok=False)
    for path in existing:
        shutil.copy2(path, destination / path.name)
    return destination


def _find_delivery_tool_dir() -> Path:
    candidates = sorted(
        (Path.home() / ".codex/plugins/cache/openai-curated-remote/data-analytics").glob(
            "*/skills/build-report/scripts/deliver_portable_artifact.mjs"
        ),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "The Data Analytics portable report builder is not installed. "
            "Use --artifact-only or install the matching report skill."
        )
    return candidates[0].parent


def _deliver_html() -> dict[str, Any]:
    command = [
        "node",
        str(PROJECT_ROOT / "scripts/package_project_explainer.mjs"),
        "--input",
        str(ARTIFACT_PATH),
        "--output",
        str(HTML_PATH),
        "--tool-dir",
        str(_find_delivery_tool_dir()),
        "--screenshot",
        str(PROJECT_ROOT / "reports/project_explainer_delivery_failure.png"),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        details = "\n".join(
            item for item in (completed.stdout.strip(), completed.stderr.strip()) if item
        )
        raise RuntimeError(f"Portable report delivery failed:\n{details}")
    receipt = json.loads(completed.stdout)
    if completed.stderr.strip():
        receipt["stderr"] = completed.stderr.strip()
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-only",
        action="store_true",
        help="Write evidence and canonical artifact JSON without packaging HTML.",
    )
    parser.add_argument(
        "--archive-existing",
        action="store_true",
        help="Copy existing explainer outputs to a UTC-dated archive before replacement.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    archived = _archive_existing() if args.archive_existing else None
    evidence = reproduce_evidence()
    _write_json(EVIDENCE_PATH, evidence)
    artifact = build_artifact(evidence)
    _write_json(ARTIFACT_PATH, artifact)
    output: dict[str, Any] = {
        "archive": None if archived is None else archived.relative_to(PROJECT_ROOT).as_posix(),
        "evidence": EVIDENCE_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "artifact": ARTIFACT_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "html": None,
    }
    if not args.artifact_only:
        output["delivery"] = _deliver_html()
        output["html"] = HTML_PATH.relative_to(PROJECT_ROOT).as_posix()
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
