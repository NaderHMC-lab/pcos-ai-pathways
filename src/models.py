"""8 model families, SMOTE pipelines, Optuna tuning."""
import warnings
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, StackingClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna

from config import RANDOM_STATE

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_preprocessor(numeric_cols, categorical_cols):
    num = SKPipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())])
    cat = SKPipeline([("impute", SimpleImputer(strategy="most_frequent"))])
    return ColumnTransformer([("num", num, numeric_cols), ("cat", cat, categorical_cols)],
                             remainder="drop")


def make_preprocessor_tree(numeric_cols, categorical_cols):
    num = SKPipeline([("impute", SimpleImputer(strategy="median"))])
    cat = SKPipeline([("impute", SimpleImputer(strategy="most_frequent"))])
    return ColumnTransformer([("num", num, numeric_cols), ("cat", cat, categorical_cols)],
                             remainder="drop")


def _pipe(preprocessor, clf):
    return ImbPipeline([("pre", preprocessor),
                        ("smote", SMOTE(random_state=RANDOM_STATE)),
                        ("clf", clf)])


def base_models(numeric_cols, categorical_cols):
    pre_scaled = make_preprocessor(numeric_cols, categorical_cols)
    pre_tree = make_preprocessor_tree(numeric_cols, categorical_cols)
    models = {
        "logreg": _pipe(pre_scaled, LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        "rf": _pipe(pre_tree, RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
        "extratrees": _pipe(pre_tree, ExtraTreesClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
        "svm_rbf": _pipe(pre_scaled, SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        "xgboost": _pipe(pre_tree, XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5, eval_metric="logloss",
            random_state=RANDOM_STATE, tree_method="hist", n_jobs=-1)),
        "lightgbm": _pipe(pre_tree, LGBMClassifier(
            n_estimators=300, learning_rate=0.05, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)),
        "catboost": _pipe(pre_tree, CatBoostClassifier(
            iterations=300, learning_rate=0.05, verbose=False, random_state=RANDOM_STATE)),
    }
    stack = StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
            ("xgb", XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5,
                                  eval_metric="logloss", random_state=RANDOM_STATE,
                                  tree_method="hist", n_jobs=-1)),
            ("lgbm", LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                    random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)),
        ],
        final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        cv=5, n_jobs=-1,
    )
    models["stacking"] = _pipe(pre_tree, stack)
    return models


def _obj_xgb(trial, X, y, num, cat, cv):
    p = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
    }
    clf = XGBClassifier(**p, eval_metric="logloss", random_state=RANDOM_STATE,
                        tree_method="hist", n_jobs=-1)
    return cross_val_score(_pipe(make_preprocessor_tree(num, cat), clf),
                           X, y, cv=cv, scoring="roc_auc", n_jobs=1).mean()


def _obj_lgbm(trial, X, y, num, cat, cv):
    p = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
    }
    clf = LGBMClassifier(**p, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
    return cross_val_score(_pipe(make_preprocessor_tree(num, cat), clf),
                           X, y, cv=cv, scoring="roc_auc", n_jobs=1).mean()


def _obj_cat(trial, X, y, num, cat, cv):
    p = {
        "iterations": trial.suggest_int("iterations", 100, 600),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 5),
    }
    clf = CatBoostClassifier(**p, verbose=False, random_state=RANDOM_STATE)
    return cross_val_score(_pipe(make_preprocessor_tree(num, cat), clf),
                           X, y, cv=cv, scoring="roc_auc", n_jobs=1).mean()


OPTUNA = {
    "xgboost": (_obj_xgb, XGBClassifier),
    "lightgbm": (_obj_lgbm, LGBMClassifier),
    "catboost": (_obj_cat, CatBoostClassifier),
}


def tune(name, X, y, num, cat, n_trials=20, cv_splits=5):
    obj, cls = OPTUNA[name]
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(lambda t: obj(t, X, y, num, cat, cv), n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    if name == "xgboost":
        clf = cls(**best, eval_metric="logloss", random_state=RANDOM_STATE,
                  tree_method="hist", n_jobs=-1)
    elif name == "lightgbm":
        clf = cls(**best, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
    else:
        clf = cls(**best, verbose=False, random_state=RANDOM_STATE)
    return _pipe(make_preprocessor_tree(num, cat), clf), best, study.best_value
