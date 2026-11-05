# How to participate

## The task

Predict the RNA-seq expression **`counts`** of a gene under a given
experimental condition, from three sources of information:

1. **gene table**: one row per gene, with its genomic coordinates, length
   and normalization offset;
2. **genome**: one GenBank `.gbff` file per genome, where the coordinates
   index into the DNA sequence (e.g. the promoter region upstream of the
   gene);
3. **run table** (`samples_df`): one row per RNA-seq run, with its curated
   experimental conditions (growth medium, temperature, pH, oxygen, growth
   phase, perturbation), empty when unknown.

Your model outputs two numbers per observation, **`mu`** (expected count,
≥ 0) and **`sigma`** (Negative-Binomial dispersion, > 0). Submissions are
ranked by the mean **Negative-Binomial negative log-likelihood** (lower is
better), with MAE and R² on `log1p(counts)` shown alongside. Each
observation's loss is capped at 100 nats (a typical one costs about 7), so a
single catastrophic prediction, such as `mu` near 0 for a highly expressed
gene, cannot dominate a region. Keep `mu` away from 0 anyway: a capped
observation still costs 100.

## The split

The data has three splits: **train**, **eval** and **test**. Each eval or
test observation falls in one of four regions:

- **A**: a gene and a run both seen in training, but never together;
- **B**: a new run (new experimental condition), for a gene seen in training;
- **C**: a new gene, in a run seen in training;
- **D**: a new gene in a new run.

The eval and test splits are drawn with the same criteria. You get the eval
counts, to score your model locally. The test counts stay on the platform,
where training also includes the eval observations: there, "seen in
training" means seen in train or eval.

The ranking uses the NB-NLL averaged over A, B and C. D is shown for
information.

Each test region is split in two halves: half of its observations (A), new
runs (B) or new genes (C, D) go to a **public** half, the others to a
**private** half. The leaderboard shows your scores on the public half. The final ranking uses the
private half, hidden until the end of the challenge, so tuning your model to
the leaderboard does not pay.

## What you submit

A single **`submission.py`** defining `class Solver(GeneCompetSolver)`. All
your Python code must live in this one file: the platform loads it on its own,
so it cannot import other `.py` files you ship. You may add non-code files
next to it (model weights, configs, precomputed tables) and read them from
`meta["submission_dir"]` in `get_model`. You implement two methods:

```python
class Solver(GeneCompetSolver):
    name = "submission"

    def get_model(self, meta):
        # build your (untrained) model
        ...

    def fit(self, model, train_df, samples_df, genomes_dir):
        # engineer features from the three sources and train
        ...
```

Your trained model must expose
`predict(test_df, samples_df, genomes_dir) -> ndarray (N, 2)` returning
`(mu, sigma)`. The platform **trains your code** on identical hardware, then
evaluates it. There, `fit` gets the train **and** the eval observations, so
your training must adapt to more data than in your local runs. See `solution/submission.py` for a working baseline
(feature builder + Ridge regression + dispersion calibration).

The platform installs nothing at submission time: `submission.py` can only
import the packages of `requirements.txt`, and a solver's benchopt
`requirements` are ignored. To use another package, open a pull request on
the challenge repository that adds it to `extra-deps.txt`.

## The compute

Each submission runs on its own AWS g4dn.2xlarge machine: one NVIDIA T4 GPU
(16 GB), 8 vCPUs and 32 GB of RAM, in the competition's Docker image
(`tommoral/bacteria-compet:v1`, CUDA 12.6). Training and evaluation together
must fit in **1 hour**, or the submission fails. The three Datacamp
challenges share these machines, so a submission may wait in the queue at
busy times.

## Get the data

One command downloads all the public data from Zenodo, about 1.1 GB: the
train and eval counts, the gene and run annotations, and the genomes.

```bash
pip install -r requirements.txt
benchopt prepare benchmark -d "BacterialExpression[mode=train]"
```

The data goes to `benchmark/data/`, or to the folder `BENCHOPT_DATA_HOME`
names. `mode=debug` downloads a small E. coli subset instead, to try things
quickly. The test counts are not public: they stay on the platform.

## Run it locally — same code as the server

```bash
# fits on train and scores on the eval regions A/B/C/D
benchopt run benchmark -d "BacterialExpression[mode=train]" \
    -s solution/submission.py
```

Locally, the eval regions follow the same protocol as the test regions of
the leaderboard. The benchmark you run *is* the
benchmark the server runs.

## Ideas to improve the baseline

Richer promoter representations (k-mers, learned sequence encoders), more
experimental metadata, a neural network (`benchmark/solvers/baseline_nn.py`), or
architectures that model gene × condition interactions.
