"""Central configuration: project-relative paths, the global seed, and the
schema contract for each of the three datasets.

Every other module imports its paths from here so that nothing in the project
depends on an absolute path or on the current working directory.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths (all derived from this file's location -> always project-relative)
# --------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INTERIM_DIR = DATA_DIR / "interim"

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
TUNING_DIR = RESULTS_DIR / "tuning"

MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

_ALL_DIRS = [
    RAW_DIR, PROCESSED_DIR, INTERIM_DIR,
    TABLES_DIR, FIGURES_DIR, TUNING_DIR,
    MODELS_DIR,
]


def ensure_dirs() -> None:
    """Create the output directories if they do not exist (raw data is never
    created here - it must be supplied by the team)."""
    for d in _ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_STATE = 42          # PDF section 7.1: set random_state=42 wherever applicable
TEST_SIZE = 0.20           # PDF section 7.2: consistent 80:20 split within a track
CV_FOLDS = 5               # PDF section 3.1: 5-fold cross-validated R^2

# --------------------------------------------------------------------------
# Dataset contracts
#
# `expected_raw_shape` is asserted softly by the audit step: a mismatch is
# reported loudly but never silently "fixed".
# --------------------------------------------------------------------------

REGRESSION = {
    "track": "regression",
    "name": "Student Performance (Multiple Linear Regression)",
    "filename": "Student_Performance.csv",
    "kaggle_ref": "nikhil7280/student-performance-multiple-linear-regression",
    "url": "https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression",
    "expected_raw_shape": (10000, 6),
    "target": "Performance Index",
    "numeric_features": [
        "Hours Studied",
        "Previous Scores",
        "Sleep Hours",
        "Sample Question Papers Practiced",
    ],
    "categorical_features": ["Extracurricular Activities"],
    "drop_columns": [],
    "synthetic": True,
    "notes": "Synthetic dataset. Target is an integer-rounded numerical score.",
}

CLASSIFICATION = {
    "track": "classification",
    "name": "Machine Predictive Maintenance Classification",
    "filename": "predictive_maintenance.csv",
    "kaggle_ref": "shivamb/machine-predictive-maintenance-classification",
    "url": "https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification",
    "expected_raw_shape": (10000, 10),
    "target": "Target",
    "positive_class": 1,           # 1 == machine failure
    "positive_class_name": "Failure",
    "negative_class_name": "No Failure",
    "numeric_features": [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ],
    "categorical_features": ["Type"],
    # UDI / Product ID are identifiers; Failure Type is a SECOND OUTCOME of the
    # same event as Target -> including it would be target leakage.
    "drop_columns": ["UDI", "Product ID", "Failure Type"],
    "leakage_columns": ["Failure Type"],
    "synthetic": True,
    "notes": (
        "Synthetic (AI4I-style) dataset. Use this Kaggle version; the UCI AI4I "
        "2020 release has a different column structure."
    ),
}

CLUSTERING = {
    "track": "clustering",
    "name": "Credit Card Customers (BankChurners)",
    "filename": "BankChurners.csv",
    "kaggle_ref": "sakshigoyal7/credit-card-customers",
    "url": "https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers",
    "expected_raw_shape": (10127, 23),
    "target": None,                       # unsupervised
    # PROVISIONAL subset - see docs/team_analysis_prompts.md Q-CL1.
    # The team must record its own justification before this is final.
    "cluster_features": [
        "Customer_Age",
        "Months_on_book",
        "Total_Relationship_Count",
        "Credit_Limit",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Avg_Utilization_Ratio",
    ],
    "feature_subset_status": "PROVISIONAL - awaiting team justification",
    "holdout_label": "Attrition_Flag",    # post-hoc validation ONLY
    "id_column": "CLIENTNUM",
    "banned_columns_prefix": "Naive_Bayes_Classifier",
    "synthetic": False,
    "notes": (
        "Attrition_Flag is held aside and must never influence feature choice, "
        "scaling, k selection, or fitting."
    ),
}

DATASETS = {
    "regression": REGRESSION,
    "classification": CLASSIFICATION,
    "clustering": CLUSTERING,
}

# --------------------------------------------------------------------------
# Plot style (PDF 7.3: colourblind-friendly palettes)
# --------------------------------------------------------------------------
PALETTE = "colorblind"
CMAP_QUALITATIVE = "tab10"
FIG_DPI = 120
