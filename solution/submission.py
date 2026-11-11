"""CatBoost submission for the bacterial gene-expression benchopt benchmark.

A submission is a benchopt solver: a ``submission.py`` defining
``class Solver(GeneCompetSolver)``.

Model
-----
CatBoost regressor on the normalized log count

    t = log1p(readcount) - (log_length_ratio + log_size_factor)

and prediction  mu = expm1(t_hat + offset)  (same form as the baseline).

Features
  * gene level, from feature_annotation + the .gbff genomes: length, strand,
    relative position, neighbour distances / strand (operon proxy), GC of the
    gene / codon position 3 / upstream windows, Shine-Dalgarno hint, start
    codon, product keywords (ribosomal, transporter, ...), genome GC / size.
  * condition level, from condition_annotation: medium, temperature, pH,
    oxygen, growth mode / phase, od600, perturbation, ...
  * out-of-fold target statistics: mean normalized expression of the same gene
    in OTHER runs, and of the same run on OTHER genes (NaN for unseen genes /
    runs). A random share of them is masked in training so the model also learns
    the "novel gene / novel species" regime that the test regions require.

Dispersion sigma is calibrated on held-out genomes, per mu bin and per regime
(gene statistic available or not), so the NB-NLL is not overconfident.
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import minimize_scalar

from benchmark_utils.base_solver import GeneCompetSolver
from benchmark_utils.nb_loss import nb_nll

try:  # biopython is only needed for sequence features
    from Bio import SeqIO
except ImportError:  # pragma: no cover
    SeqIO = None

UPSTREAM = 200
_COMP = str.maketrans("ACGT", "TGCA")
KEYWORDS = [
    "ribosomal", "hypothetical", "transposase", "transport", "trna",
    "synthase", "dehydrogenase", "regulator", "phage", "flagell",
    "chaperone", "elongation", "polymerase", "membrane", "kinase",
    "toxin", "pil", "fimbri", "crispr", "porin", "oxidoreductase",
    "hydrolase", "permease", "atp", "dna", "rna", "iron", "nitrate",
]
COND_NUM = ["temperature_celsius", "ph", "agitation_rpm", "od600"]
GROUP_COLS = [
    "experiment_accession", "medium_category", "oxygen_condition",
    "growth_phase", "perturbation_category",
]
COND_CAT = [
    "medium_category", "medium_name", "oxygen_condition", "growth_mode",
    "growth_phase", "is_wild_type", "has_perturbation",
    "perturbation_category", "perturbation_compound", "condition_confidence",
]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _key(df, col):
    return (
        df["genome_id"].astype(str) + "|" + df[col].astype(str)
    ).to_numpy()


def _offset(df):
    n = len(df)
    llr = (
        df["log_length_ratio"].to_numpy(dtype=float)
        if "log_length_ratio" in df.columns else np.zeros(n)
    )
    lsf = (
        df["log_size_factor"].to_numpy(dtype=float)
        if "log_size_factor" in df.columns else np.zeros(n)
    )
    return np.nan_to_num(llr + lsf)


def _oof_mean(group_code, fold, v, n_folds):
    """Mean of v per group, excluding rows of the same (group, fold) cell."""
    n_groups = int(group_code.max()) + 1
    cell = group_code * n_folds + fold
    s_c = np.bincount(cell, v, n_groups * n_folds)
    n_c = np.bincount(cell, None, n_groups * n_folds).astype(float)
    s_t = s_c.reshape(n_groups, n_folds).sum(1)
    n_t = n_c.reshape(n_groups, n_folds).sum(1)
    s = s_t[group_code] - s_c[cell]
    n = n_t[group_code] - n_c[cell]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return mean, n


def _gc(s):
    return (s.count("G") + s.count("C")) / len(s) if len(s) else np.nan


def _frac(s, c):
    return s.count(c) / len(s) if len(s) else np.nan


def _first_num(series, last=False):
    parts = series.astype(str).str.split(";")
    part = parts.str[-1] if last else parts.str[0]
    return pd.to_numeric(part, errors="coerce")


def _fit_sigma(y, mu):
    res = minimize_scalar(
        lambda lr: nb_nll(y, mu, np.exp(lr)),
        bounds=(-4.0, 8.0), method="bounded",
    )
    return float(np.exp(res.x))


def _scale_curve(y, mu, s0, n_bins=6):
    """log mean-correction as a function of log1p(mu): (xs, log_c)."""
    lm = np.log1p(mu)
    if len(y) < 800:
        res = minimize_scalar(
            lambda lc: nb_nll(y, np.maximum(mu * np.exp(lc), 0.1), s0),
            bounds=(-1.5, 1.5), method="bounded",
        )
        return np.array([0.0, 1.0]), np.array([res.x, res.x])
    edges = np.quantile(lm, np.linspace(0, 1, n_bins + 1))
    xs, ls = [], []
    for i in range(n_bins):
        hi = (lm <= edges[i + 1]) if i == n_bins - 1 else (lm < edges[i + 1])
        m = (lm >= edges[i]) & hi
        if m.sum() < 150:
            continue
        res = minimize_scalar(
            lambda lc: nb_nll(
                y[m], np.maximum(mu[m] * np.exp(lc), 0.1), s0),
            bounds=(-1.5, 1.5), method="bounded",
        )
        xs.append(np.median(lm[m]))
        ls.append(res.x)
    if not xs:
        return np.array([0.0, 1.0]), np.zeros(2)
    xs, idx = np.unique(np.array(xs), return_index=True)
    return xs, np.array(ls)[idx]


def _sigma_curve(y, mu, n_bins=8):
    """sigma as a function of log1p(mu): (x_grid, log_sigma_grid)."""
    glob = _fit_sigma(y, mu)
    lm = np.log1p(mu)
    if len(y) < 800:
        return np.array([0.0, 1.0]), np.log([glob, glob])
    edges = np.quantile(lm, np.linspace(0, 1, n_bins + 1))
    xs, ls = [], []
    for i in range(n_bins):
        hi = (lm <= edges[i + 1]) if i == n_bins - 1 else (lm < edges[i + 1])
        m = (lm >= edges[i]) & hi
        if m.sum() < 100:
            continue
        xs.append(np.median(lm[m]))
        ls.append(np.log(_fit_sigma(y[m], mu[m])))
    if not xs:
        return np.array([0.0, 1.0]), np.log([glob, glob])
    xs, idx = np.unique(np.array(xs), return_index=True)
    return xs, np.array(ls)[idx]


# ----------------------------------------------------------------------------
# gene-level features from annotation + genome
# ----------------------------------------------------------------------------
def _genome_table(path, g):
    """Feature table (indexed by genome|feature_id) for one genome."""
    gid = str(g["genome_id"].iloc[0])
    chrom = g["Chr"].astype(str).str.split(";").str[0].to_numpy()
    start = _first_num(g["Start"]).to_numpy(dtype=float)
    end = _first_num(g["End"], last=True).to_numpy(dtype=float)
    strand = g["Strand"].astype(str).str.split(";").str[0].to_numpy()
    fids = g["feature_id"].astype(str).to_numpy()
    n = len(g)

    seqs, prods = {}, {}
    genome_gc, genome_len = np.nan, np.nan
    if SeqIO is not None and path is not None and path.exists():
        tot_gc = tot_len = 0
        for rec in SeqIO.parse(str(path), "genbank"):
            s = str(rec.seq).upper()
            seqs[rec.id] = s
            seqs[rec.name] = s
            tot_gc += s.count("G") + s.count("C")
            tot_len += len(s)
            for f in rec.features:
                if f.type != "CDS":
                    continue
                pid = f.qualifiers.get("protein_id", [None])[0]
                if pid and pid not in prods:
                    prods[pid] = f.qualifiers.get("product", [""])[0].lower()
        if tot_len:
            genome_gc, genome_len = tot_gc / tot_len, tot_len

    cols = [
        "gc_body", "gc3", "gc_p50", "gc_p100", "gc_p200", "a_p100", "t_p100",
        "sd", "start_codon", "rel_pos",
    ]
    out = np.full((n, len(cols)), np.nan, dtype=np.float32)
    kw = np.zeros((n, len(KEYWORDS)), dtype=np.float32)
    kw_known = np.zeros(n, dtype=np.float32)

    for i in range(n):
        s = seqs.get(chrom[i])
        if s is None or np.isnan(start[i]) or np.isnan(end[i]):
            continue
        a, b = int(start[i]), int(end[i])
        plus = strand[i] != "-"
        if plus:
            body = s[a - 1:b]
            prom = s[max(0, a - 1 - UPSTREAM):a - 1]
        else:
            body = s[a - 1:b].translate(_COMP)[::-1]
            prom = s[b:b + UPSTREAM].translate(_COMP)[::-1]
        if not body:
            continue
        p100 = prom[-100:]
        tail = prom[-20:]
        sc = body[:3]
        out[i] = (
            _gc(body), _gc(body[2::3]), _gc(prom[-50:]), _gc(p100), _gc(prom),
            _frac(p100, "A"), _frac(p100, "T"),
            float("GGAG" in tail or "AGGA" in tail or "GAGG" in tail),
            {"ATG": 0, "GTG": 1, "TTG": 2}.get(sc, 3),
            (a + b) / 2 / len(s),
        )
        prod = prods.get(fids[i].split("cds-", 1)[-1])
        if prod is not None:
            kw_known[i] = 1.0
            for j, w in enumerate(KEYWORDS):
                if w in prod:
                    kw[i, j] = 1.0

    tab = pd.DataFrame(out, columns=cols)
    tab[[f"kw_{w}" for w in KEYWORDS]] = kw
    tab["kw_known"] = kw_known

    length = pd.to_numeric(g["Length"], errors="coerce").to_numpy(dtype=float)
    tab["log_len"] = np.log(np.maximum(length, 1.0))
    tab["strand_plus"] = (strand != "-").astype(np.float32)
    tab["on_edge"] = (
        g["on_contig_edge"].astype(str).str.lower().eq("true")
        .to_numpy(dtype=np.float32)
        if "on_contig_edge" in g.columns else 0.0
    )
    tab["genome_gc"] = genome_gc
    tab["genome_logsize"] = np.log(genome_len) if genome_len == genome_len \
        else np.nan
    tab["genome_ngenes"] = np.log(n)

    # neighbour layout (operon proxy)
    lay = pd.DataFrame(
        {"chr": chrom, "s": start, "e": end, "p": tab["strand_plus"]}
    ).sort_values(["chr", "s"])
    grp = lay.groupby("chr", sort=False)
    lay["d_prev"] = lay["s"] - grp["e"].shift(1)
    lay["d_next"] = grp["s"].shift(-1) - lay["e"]
    lay["same_prev"] = (lay["p"] == grp["p"].shift(1)).astype(np.float32)
    lay["same_next"] = (lay["p"] == grp["p"].shift(-1)).astype(np.float32)
    lay = lay.sort_index()
    tab["d_prev"] = lay["d_prev"].clip(-100, 5000).to_numpy(np.float32)
    tab["d_next"] = lay["d_next"].clip(-100, 5000).to_numpy(np.float32)
    tab["same_prev"] = lay["same_prev"].to_numpy()
    tab["same_next"] = lay["same_next"].to_numpy()

    tab.index = gid + "|" + pd.Index(fids)
    return tab[~tab.index.duplicated()]


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
class SubmissionModel:

    def __init__(
        self,
        meta=None,
        iterations=1500,
        learning_rate=0.08,
        depth=8,
        max_rows=1_500_000,
        mask_frac=0.3,
        val_frac=0.1,
        n_folds=5,
        seed=0,
        verbose=100,
    ):
        self.meta = meta or {}
        self.genes = self.meta.get("feature_annotations")
        self.params = dict(
            iterations=iterations, learning_rate=learning_rate, depth=depth,
        )
        self.max_rows = max_rows
        self.mask_frac = mask_frac
        self.val_frac = val_frac
        self.n_folds = n_folds
        self.seed = seed
        self.verbose = verbose
        self._parts = {}
        self._tab = None
        self.model = None
        self.group_specs = []

    # -- gene table ----------------------------------------------------------
    def _ensure_genes(self, genome_ids, genomes_dir):
        todo = [g for g in genome_ids if g not in self._parts]
        if not todo:
            return
        if self.genes is None:
            raise RuntimeError("meta['feature_annotations'] is required")
        ann = self.genes[self.genes["genome_id"].astype(str).isin(todo)]
        for k, (gid, g) in enumerate(ann.groupby(
                ann["genome_id"].astype(str), sort=False)):
            path = (
                genomes_dir / f"{gid}.gbff" if genomes_dir is not None
                else None
            )
            self._parts[gid] = _genome_table(path, g)
            if self.verbose and (k + 1) % 25 == 0:
                print(f"[submission] genome features {k + 1}/{len(todo)}",
                      flush=True)
        for gid in todo:  # genomes absent from the annotation
            self._parts.setdefault(gid, None)
        parts = [p for p in self._parts.values() if p is not None]
        self._tab = pd.concat(parts) if parts else pd.DataFrame()

    @staticmethod
    def _run_values(samples_df, rkeys, col):
        sd = samples_df.copy()
        sd["_rk"] = _key(sd, "run_accession")
        sd = sd.drop_duplicates("_rk").set_index("_rk")
        v = sd[col].reindex(rkeys)
        return v.astype(object).where(v.notna(), "NA").astype(str).to_numpy()

    # -- design matrix ---------------------------------------------------------
    def _design(self, obs, samples_df, genomes_dir, st):
        self._ensure_genes(
            list(obs["genome_id"].astype(str).unique()), genomes_dir
        )
        gk, rk = _key(obs, "feature_id"), _key(obs, "run_accession")
        X = pd.DataFrame({
            "log_length_ratio": obs["log_length_ratio"].to_numpy(float)
            if "log_length_ratio" in obs.columns else 0.0,
            "log_size_factor": obs["log_size_factor"].to_numpy(float)
            if "log_size_factor" in obs.columns else 0.0,
        })
        if self._tab is not None and len(self._tab.columns):
            gt = self._tab.reindex(gk)
            gt.index = X.index
            X = pd.concat([X, gt], axis=1)

        cat_cols = []
        if samples_df is not None:
            sd = samples_df.copy()
            sd["_rk"] = _key(sd, "run_accession")
            sd = sd.drop_duplicates("_rk").set_index("_rk")
            rt = sd.reindex(rk)
            rt.index = X.index
            for c in COND_NUM:
                if c in rt.columns:
                    X[c] = pd.to_numeric(rt[c], errors="coerce")
            for c in COND_CAT:
                if c in rt.columns:
                    X[c] = rt[c].astype(object).where(rt[c].notna(), "NA") \
                        .astype(str)
                    cat_cols.append(c)

        mu0 = self.mu0
        for k, v in st.items():
            is_count = k in ("g_n", "r_n") or k.startswith("gn_")
            X[k] = np.log1p(v) if is_count else v
        X["add_pred"] = (
            np.where(np.isnan(st["g_mean"]), mu0, st["g_mean"])
            + np.where(np.isnan(st["r_mean"]), mu0, st["r_mean"]) - mu0
        )
        for c in X.columns:
            if c not in cat_cols:
                X[c] = X[c].astype(np.float32)
        self.cat_cols = cat_cols
        return X

    # -- calibration -------------------------------------------------------------
    @staticmethod
    def _calibrate(y, mu_raw):
        s0 = _fit_sigma(y, mu_raw)
        sc = _scale_curve(y, mu_raw, s0)
        mu_cal = np.maximum(
            mu_raw * np.exp(np.interp(np.log1p(mu_raw), *sc)), 0.1)
        return {"scale": sc, "sigma": _sigma_curve(y, mu_cal)}

    # -- fit ---------------------------------------------------------------------
    def fit(self, train_df, samples_df=None, genomes_dir=None):
        rng = np.random.default_rng(self.seed)
        target_col = "readcount" if "readcount" in train_df.columns \
            else "counts"
        train_df = train_df.reset_index(drop=True)
        y = train_df[target_col].to_numpy(dtype=float)
        offset = _offset(train_df)
        t = np.log1p(y) - offset
        self.mu0 = float(np.mean(t))

        gk, rk = _key(train_df, "feature_id"), _key(train_df, "run_accession")
        gcode, guniq = pd.factorize(gk)
        rcode, runiq = pd.factorize(rk)
        K = self.n_folds
        run_fold = rng.integers(0, K, len(runiq))
        gene_fold = rng.integers(0, K, len(guniq))
        rfold = run_fold[rcode]

        # full per-run mean -> run-depth-free target
        rc_n = np.bincount(rcode).astype(float)
        run_m = np.bincount(rcode, t) / rc_n
        t_adj = t - run_m[rcode]

        # out-of-fold statistics (the row's own run never contributes)
        g_mean, g_n = _oof_mean(gcode, rfold, t, K)
        m2, _ = _oof_mean(gcode, rfold, t * t, K)
        st = {
            "g_mean": g_mean,
            "g_n": g_n,
            "g_res": _oof_mean(gcode, rfold, t_adj, K)[0],
            "g_std": np.sqrt(np.maximum(m2 - g_mean ** 2, 0.0)),
        }
        r_mean, r_n = _oof_mean(rcode, gene_fold[gcode], t, K)

        # gene x condition-group statistics (replicates, same medium, ...)
        self.group_specs = []
        gst = {}
        if samples_df is not None:
            for col in GROUP_COLS:
                if col not in samples_df.columns:
                    continue
                vals = self._run_values(samples_df, runiq, col)
                rg_code, rg_index = pd.factorize(vals)
                n_rg = len(rg_index)
                comb = gcode.astype(np.int64) * n_rg + rg_code[rcode]
                uniq, inv = np.unique(comb, return_inverse=True)
                inv = inv.ravel()
                gm, gn = _oof_mean(inv, rfold, t, K)
                cnt = np.bincount(inv).astype(float)
                tab = pd.DataFrame(
                    {"m": np.bincount(inv, t) / cnt, "n": cnt}, index=uniq)
                if col == "experiment_accession":  # emulate new experiments
                    em = (rng.random(n_rg) < self.mask_frac)[rg_code[rcode]]
                    gm[em], gn[em] = np.nan, 0.0
                gst[f"gm_{col}"], gst[f"gn_{col}"] = gm, gn
                self.group_specs.append(
                    dict(col=col, rg_index=pd.Index(rg_index), n_rg=n_rg,
                         tab=tab))
        st.update(gst)

        # full statistics used at prediction time
        gn_full = np.bincount(gcode).astype(float)
        self.g_index = pd.Index(guniq)
        self.gene_stats = pd.DataFrame(
            {
                "m": np.bincount(gcode, t) / gn_full,
                "n": gn_full,
                "res": np.bincount(gcode, t_adj) / gn_full,
                "m2": np.bincount(gcode, t * t) / gn_full,
            },
            index=guniq,
        )
        self.run_stats = pd.DataFrame({"m": run_m, "n": rc_n}, index=runiq)

        # mask a share of genes / runs: emulate novel genes / species
        gmask = (rng.random(len(guniq)) < self.mask_frac)[gcode]
        rmask = (rng.random(len(runiq)) < self.mask_frac)[rcode]
        for k in list(st):
            if k.startswith(("g_", "gm_", "gn_")):
                st[k] = np.where(gmask, 0.0 if k.endswith("_n") or
                                 k.startswith("gn_") else np.nan, st[k])
        st["r_mean"] = np.where(rmask, np.nan, r_mean)
        st["r_n"] = np.where(rmask, 0.0, r_n)

        # subsample rows
        n = len(train_df)
        sel = np.sort(rng.choice(n, min(n, self.max_rows), replace=False))
        df = train_df.iloc[sel].reset_index(drop=True)
        t, offset, y = t[sel], offset[sel], y[sel]
        st = {k: v[sel] for k, v in st.items()}

        X = self._design(df, samples_df, genomes_dir, st)
        self.cols = list(X.columns)

        # validation split by genome (by gene when there are few genomes)
        grp = df["genome_id"].astype(str).to_numpy()
        if len(np.unique(grp)) < 5:
            grp = gk[sel]
        ug = np.unique(grp)
        n_val = max(1, int(round(self.val_frac * len(ug))))
        va = np.isin(grp, rng.choice(ug, n_val, replace=False))
        tr = ~va

        self.model = CatBoostRegressor(
            loss_function="RMSE",
            od_type="Iter",
            od_wait=100,
            thread_count=-1,
            random_seed=self.seed,
            max_ctr_complexity=1,
            verbose=self.verbose,
            allow_writing_files=False,
            **self.params,
        )
        self.model.fit(
            X[tr], t[tr], cat_features=self.cat_cols,
            eval_set=(X[va], t[va]), use_best_model=True,
        )

        # calibrate mean and dispersion on held-out genomes, per regime
        mu_raw = self._mu(self.model.predict(X[va]), offset[va])
        yv = y[va]
        regime = (st["g_n"][va] > 0).astype(int)
        glob = self._calibrate(yv, mu_raw)
        self.calib = {}
        for r in (0, 1):
            m = regime == r
            self.calib[r] = (
                self._calibrate(yv[m], mu_raw[m]) if m.sum() >= 2000 else glob
            )
        return self

    # -- predict -----------------------------------------------------------------
    @staticmethod
    def _mu(pred, offset):
        return np.maximum(np.expm1(np.clip(pred + offset, -15.0, 15.0)), 0.1)

    def _predict_stats(self, gk, rk, samples_df):
        gs = self.gene_stats.reindex(gk)
        rs = self.run_stats.reindex(rk)
        g_mean = gs["m"].to_numpy(float)
        st = {
            "g_mean": g_mean,
            "g_n": np.nan_to_num(gs["n"].to_numpy(float)),
            "g_res": gs["res"].to_numpy(float),
            "g_std": np.sqrt(np.maximum(
                gs["m2"].to_numpy(float) - g_mean ** 2, 0.0)),
            "r_mean": rs["m"].to_numpy(float),
            "r_n": np.nan_to_num(rs["n"].to_numpy(float)),
        }
        gcode_t = self.g_index.get_indexer(gk)
        rcode_t, runiq_t = pd.factorize(rk)
        for spec in self.group_specs:
            col = spec["col"]
            if samples_df is None or col not in samples_df.columns:
                m = np.full(len(gk), np.nan)
                n = np.zeros(len(gk))
            else:
                vals = self._run_values(samples_df, runiq_t, col)
                rg = spec["rg_index"].get_indexer(vals)[rcode_t]
                comb = np.where(
                    (gcode_t >= 0) & (rg >= 0),
                    gcode_t.astype(np.int64) * spec["n_rg"] + rg, -1)
                tb = spec["tab"].reindex(comb)
                m = tb["m"].to_numpy(float)
                n = np.nan_to_num(tb["n"].to_numpy(float))
            st[f"gm_{col}"], st[f"gn_{col}"] = m, n
        return st

    def predict(self, test_df, samples_df=None, genomes_dir=None):
        test_df = test_df.reset_index(drop=True)
        n = len(test_df)
        offset = _offset(test_df)
        gk, rk = _key(test_df, "feature_id"), _key(test_df, "run_accession")
        st = self._predict_stats(gk, rk, samples_df)

        X = self._design(test_df, samples_df, genomes_dir, st)
        for c in self.cols:  # safety: same columns / order as in fit
            if c not in X.columns:
                X[c] = np.nan
        X = X[self.cols]

        mu_raw = self._mu(self.model.predict(X), offset)
        regime = (st["g_n"] > 0).astype(int)
        mu = np.empty(n)
        sigma = np.empty(n)
        for r in (0, 1):
            m = regime == r
            if not m.any():
                continue
            cal = self.calib[r]
            lm = np.log1p(mu_raw[m])
            mu_r = np.maximum(
                mu_raw[m] * np.exp(np.interp(lm, *cal["scale"])), 0.1)
            xs, ls = cal["sigma"]
            mu[m] = mu_r
            sigma[m] = np.exp(np.interp(np.log1p(mu_r), xs, ls))
        out = np.column_stack([mu, np.clip(sigma, 0.02, 1e4)])
        return np.nan_to_num(out, nan=1.0, posinf=1e4)


class Solver(GeneCompetSolver):

    name = "submission"

    def get_model(self, meta):
        return SubmissionModel(meta)

    def fit(self, model, train_df, samples_df, genomes_dir):
        model.fit(train_df, samples_df, genomes_dir)
