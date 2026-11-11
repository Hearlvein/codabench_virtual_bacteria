"""Sample submission — the starting point you copy and improve.

A submission is a benchopt solver: a ``submission.py`` defining
``class Solver(GeneCompetSolver)``.

This baseline decomposes normalized log-expression into marginal additive
effects per gene and per condition, then calibrates a single Negative-Binomial
dispersion sigma on the training set:
    log1p(mu_{g, s}) = mu_0 + alpha_g + beta_s + offset_{g, s}
where offset_{g, s} = log_size_factor_s + log_length_ratio_g.

Run it locally exactly as the platform does::

    python tools/setup_data.py
    BENCHOPT_DATA_HOME=phases/main/input_data/data \\
        benchopt run benchmark -d BacterialExpression -s solution/submission.py

Ideas to improve it:
- Gradient boosting (e.g. CatBoost, LightGBM) on gene x condition interactions
- Sequence representations from genomes (promoter k-mers, learned encoders)
- Graph neural networks leveraging the homology & proximity graph
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from benchmark_utils.base_solver import GeneCompetSolver
from benchmark_utils.nb_loss import nb_nll


class SubmissionModel:

    def __init__(self, shrinkage: float = 1.0):
        self.shrinkage = float(shrinkage)
        self.mu_0 = 0.0
        self.gene_effects = {}
        self.cond_effects = {}
        self.sigma = 1.0

    def fit(self, train_df: pd.DataFrame, samples_df=None, genomes_dir=None):
        target_col = (
            "readcount" if "readcount" in train_df.columns else "counts"
        )
        y = train_df[target_col].to_numpy(dtype=float)

        llr = (
            train_df["log_length_ratio"].to_numpy(dtype=float)
            if "log_length_ratio" in train_df.columns
            else np.zeros(len(train_df))
        )
        lsf = (
            train_df["log_size_factor"].to_numpy(dtype=float)
            if "log_size_factor" in train_df.columns
            else np.zeros(len(train_df))
        )
        offset = llr + lsf

        # Composite IDs
        if "gene_uid" in train_df.columns:
            gene_ids = train_df["gene_uid"].astype(str).values
        else:
            gene_ids = (
                train_df["genome_id"].astype(str)
                + "_"
                + train_df["feature_id"].astype(str)
            ).values

        if "condition_uid" in train_df.columns:
            cond_ids = train_df["condition_uid"].astype(str).values
        else:
            cond_ids = (
                train_df["genome_id"].astype(str)
                + "_"
                + train_df["run_accession"].astype(str)
            ).values

        # Normalized log counts
        log_y = np.log1p(y) - offset
        self.mu_0 = float(np.mean(log_y))
        y_centered = log_y - self.mu_0

        # Gene effects with shrinkage
        lam = self.shrinkage
        df_genes = pd.DataFrame({"gene": gene_ids, "val": y_centered})
        g_stats = df_genes.groupby("gene")["val"].agg(["sum", "count"])
        self.gene_effects = (
            g_stats["sum"] / (g_stats["count"] + lam)
        ).to_dict()

        # Condition effects with shrinkage
        alpha_vec = np.array([self.gene_effects.get(g, 0.0) for g in gene_ids])
        resid = y_centered - alpha_vec
        df_conds = pd.DataFrame({"cond": cond_ids, "val": resid})
        c_stats = df_conds.groupby("cond")["val"].agg(["sum", "count"])
        self.cond_effects = (
            c_stats["sum"] / (c_stats["count"] + lam)
        ).to_dict()

        # Calibrate dispersion sigma
        beta_vec = np.array([self.cond_effects.get(c, 0.0) for c in cond_ids])
        log_pred = self.mu_0 + alpha_vec + beta_vec + offset
        mu_fit = np.maximum(np.expm1(np.clip(log_pred, -15.0, 15.0)), 0.5)

        res = minimize_scalar(
            lambda log_r: nb_nll(y, mu_fit, np.exp(log_r)),
            bounds=(-4.0, 8.0),
            method="bounded",
        )
        self.sigma = float(np.exp(res.x))
        return self

    def predict(
        self, test_df: pd.DataFrame, samples_df=None, genomes_dir=None
    ):
        n = len(test_df)
        llr = (
            test_df["log_length_ratio"].to_numpy(dtype=float)
            if "log_length_ratio" in test_df.columns
            else np.zeros(n)
        )
        lsf = (
            test_df["log_size_factor"].to_numpy(dtype=float)
            if "log_size_factor" in test_df.columns
            else np.zeros(n)
        )
        offset = llr + lsf

        if "gene_uid" in test_df.columns:
            gene_ids = test_df["gene_uid"].astype(str).values
        else:
            gene_ids = (
                test_df["genome_id"].astype(str)
                + "_"
                + test_df["feature_id"].astype(str)
            ).values

        if "condition_uid" in test_df.columns:
            cond_ids = test_df["condition_uid"].astype(str).values
        else:
            cond_ids = (
                test_df["genome_id"].astype(str)
                + "_"
                + test_df["run_accession"].astype(str)
            ).values

        alpha = np.array([self.gene_effects.get(g, 0.0) for g in gene_ids])
        beta = np.array([self.cond_effects.get(c, 0.0) for c in cond_ids])

        log_pred = self.mu_0 + alpha + beta + offset
        mu = np.maximum(np.expm1(np.clip(log_pred, -15.0, 15.0)), 0.5)
        return np.column_stack([mu, np.full(n, self.sigma)])


class Solver(GeneCompetSolver):

    name = "submission"

    def get_model(self, meta):
        return SubmissionModel(shrinkage=1.0)

    def fit(self, model, train_df, samples_df, genomes_dir):
        model.fit(train_df, samples_df, genomes_dir)
