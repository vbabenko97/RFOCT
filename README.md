# RFOCT

RFOCT (Random Forest of Optimal-Complexity Trees) is a custom tree ensemble for
classification. This repository is the research/reference implementation of the
published algorithm; it does **not** wrap or substitute
`sklearn.ensemble.RandomForestClassifier`.

The implementation supports binary and native multiclass classification through a
scikit-learn-compatible `RFOCTClassifier`. It preserves the final research code's core
algorithmic behavior while packaging it for deterministic, testable use.

> [!IMPORTANT]
> This is research software, currently version `0.1.0`. Validate it independently before
> using it for clinical, safety-critical, or production decisions.

## Algorithm

An RFOCT forest combines:

- bootstrap sampling and out-of-bag (OOB) evaluation;
- binary-split optimal-complexity trees built with an internal A/B protocol: candidate
  thresholds are fitted on A and the split feature is selected on B;
- MCC or Gini split objectives;
- genetic-algorithm selection of each tree's feature subset, scored on OOB samples;
- uniform or quality-weighted tree voting;
- native multiclass leaves and class voting, while retaining binary tree splits.

Feature-subset selection and weighting use training-derived OOB signals. Held-out test data
is not accepted by `fit` and is not used for model selection.

### AHP-style weighted voting

The implementation uses a dense quality-level convention derived from tree OOB scores. The
weakest distinct score receives quality level `1`; levels increase with score, and tied scores
receive equal levels. For levels `q_i`, the pairwise matrix is

```text
a_ij = q_i / q_j
```

Normalized row geometric means form the voting weights. Therefore, stronger estimators get
larger weights. This convention intentionally differs from rank systems where `1` means
"best". Historical prose used that inverse best-rank terminology, but it conflicts with both
the executable legacy code and archived-model weights. The reference implementation therefore
treats the executable priority-level convention as the compatibility assumption.

Binary prediction thresholds the weighted `0`/`1` vote share at `0.5`; multiclass prediction
selects the class with the largest accumulated vote weight.

## Installation

RFOCT requires Python 3.10 or newer. Install the current checkout with pip:

```bash
python -m pip install .
```

With [uv](https://docs.astral.sh/uv/), create a development environment with:

```bash
uv sync --extra dev
```

The runtime dependency set is limited to NumPy and scikit-learn.

## Minimal example

```python
from sklearn.datasets import make_classification

from rfoct import RFOCTClassifier

X, y = make_classification(
    n_samples=100,
    n_features=6,
    n_informative=4,
    n_redundant=0,
    random_state=7,
)

classifier = RFOCTClassifier(
    n_estimators=3,
    max_level=2,
    max_features=2,
    ga_population=2,
    ga_epochs=1,
    random_state=7,
)
classifier.fit(X, y)

predictions = classifier.predict(X[:5])
vote_shares = classifier.predict_proba(X[:5])
```

`predict_proba` returns normalized weighted vote shares in `classes_` order. These values are
not calibrated posterior probabilities.

Complete deterministic binary and multiclass examples are in
[`examples/binary.py`](examples/binary.py) and
[`examples/multiclass.py`](examples/multiclass.py).

## Estimator API

`RFOCTClassifier` follows the scikit-learn estimator parameter convention and exposes:

- `fit`, `predict`, `predict_proba`, and inherited `score`;
- `get_params` and `set_params` through scikit-learn's `BaseEstimator`;
- fitted `classes_` and `n_features_in_` attributes;
- `voting="weighted"` or `voting="uniform"`;
- `split_criterion="mcc"` or `split_criterion="gini"`;
- JSON serialization through `save`, `load`, `to_dict`, and `from_dict`.

Only load model files from trusted sources, and treat serialized models as tied to the RFOCT
version that created them unless compatibility is explicitly documented.

## Lineage

RFOCT evolved from the earlier Self-Organization Forest prototype. SOF and RFOCT are not
equivalent implementations: the final RFOCT adds bootstrap/OOB processing, GA feature
selection, optimal-complexity tree construction, weighted voting, multiclass classification,
and a scikit-learn-style estimator API.

The earlier SOF remains available in Git history:

- [last refactored SOF snapshot (`1ca0a6f`)](https://github.com/vbabenko97/Self-Organization-Forest/tree/1ca0a6f5cc1deb879382a32e5440135be66f8b15)
- [original uploaded SOF implementation (`64b6b0d`)](https://github.com/vbabenko97/Self-Organization-Forest/tree/64b6b0d3ccaf597ec28b8264a73c569dba63dcf1)

## Publication and citation

The algorithm was published as:

V. Babenko, Ie. Nastenko, V. Pavlov, O. Horodetska, I. Dykan, B. Tarasiuk, and
V. Lazoryshinets, “Classification of Pathologies on Medical Images Using the Algorithm of
Random Forest of Optimal-Complexity Trees,” *Cybernetics and Systems Analysis*, vol. 59,
no. 2, pp. 346-358, 2023.
[doi:10.1007/s10559-023-00569-z](https://doi.org/10.1007/s10559-023-00569-z)

```bibtex
@article{Babenko2023RFOCT,
  author  = {Babenko, V. and Nastenko, Ie. and Pavlov, V. and Horodetska, O. and
             Dykan, I. and Tarasiuk, B. and Lazoryshinets, V.},
  title   = {Classification of Pathologies on Medical Images Using the Algorithm of
             Random Forest of Optimal-Complexity Trees},
  journal = {Cybernetics and Systems Analysis},
  year    = {2023},
  volume  = {59},
  number  = {2},
  pages   = {346--358},
  doi     = {10.1007/s10559-023-00569-z}
}
```

Machine-readable citation metadata is provided in [`CITATION.cff`](CITATION.cff).

## Limitations

- RFOCT is computationally heavier than conventional random forests because tree induction
  performs internal A/B validation and GA feature-subset search.
- `predict_proba` reports ensemble vote shares, not calibrated probabilities.
- Native multiclass prediction uses binary-split trees and weighted class voting.
- Reproducibility with a fixed `random_state` is tested within a fixed RFOCT and dependency
  environment; cross-version bit-for-bit reproducibility is not promised.
- No benchmark claims or bundled medical/research datasets are provided by this repository.
- Applicability of the repository's existing license to the legacy-derived RFOCT code has not
  yet been established. Resolve code ownership and licensing before publishing a package or
  release; this is a release blocker.

## Development

Install development dependencies and run verification:

```bash
uv sync --extra dev
uv run ruff format --check .
uv run ruff check .
uv run pytest
uv run python examples/binary.py
uv run python examples/multiclass.py
```

Tests and examples use only small synthetic datasets. Local research data, trained models,
reports, and other artifacts are excluded from version control and package contents.
