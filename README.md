# Svm from scratch

I fit soft-margin SVMs from scratch on public UCI data, and I record every pair update the optimizer accepts. You can put prediction accuracy next to the optimization gaps, inspect exact linear proofs, and edit finite input boxes.

[Live 30-feature margin inspector](https://midya8.github.io/svm-from-scratch/) · [Exact linear proof inspector](https://midya8.github.io/svm-from-scratch/certificate.html) · [Exact input boxes](https://midya8.github.io/svm-from-scratch/box.html) · [Equations, protocol and limits](METHOD.md)

This is soft-margin SVM learning built from the ground up, on licensed public UCI data. The updates work on two variables at a time and preserve a linear equality plus the box constraints; every fit exposes its primal and dual values, the numerical KKT and feasibility residuals, and every pair it accepted. NumPy supplies the arrays; it is not an SVM solver. No sklearn SVM, no model API, no private dataset, no diagnostic product.

> Educational historical-data engineering exercise only. Not clinical validation, medical advice, calibrated probabilities or deployment readiness. Original identifiers are excluded from features and browser data.

## How it works

Linear and RBF kernels, analytic bounded pair updates, explicit handling for zero-curvature endpoints, full replay of every pair, and separate numerical checks for the gap and for feasibility. Eight fits run on a frozen 341/113/115 class-stratified split: scaling from training data only, selection from validation only, no refitting on validation, and every candidate stays visible.

The browser side does its own scalar JavaScript inference across all 30 editable features, with signed feature and kernel terms, honest two-feature decision slices, and inputs in original units. The controls cover accuracy against optimization, zero curvature, and nonunique duplicate support. And there's an independent exact-rational active-face KKT oracle for 60 tiny problems. The full-data floating-point gaps stay numerical, while separate repaired exact witnesses cover the represented linear problems.

## What came out

Validation picks linear C=1: **112/115** test labels right. A full-30D 1NN gets **111/115**; guessing the training majority gets **72/115**. RBF at C=10 and C=100 each get **113/115**, but they weren't picked on test scores. One small historical split, so this is not a general model ranking and definitely not a clinical evaluation. Every original candidate is still there.

![All eight models and both controls on the same zero-based test accuracy scale](assets/test-accuracy.svg)

All eight fits meet both explicit numerical KKT/gap tolerances. A norm/margin score isn't a probability, and a small optimization gap doesn't bound clinical or generalization error. And where examples are duplicated, different support coefficients can give the same optimum, so alpha isn't unique causal data importance.

## Running it

Python 3.10+; NumPy pinned to 2.2.6. The runtime is offline. The original data and the losslessly encoded source metadata, with hashes, are bundled under CC BY 4.0; the code is MIT.

```sh
python -m pip install -e .
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
python -m marginlab report
python -m marginlab audit
python -m marginlab fit --output fresh.json
python -m marginlab audit --report fresh.json
python -m marginlab controls
node --test tests/*.cjs
```

The actual Chrome checks need Chrome and Node 22+. Set `CHROME_BIN` if it isn't on the path. The independent checks cover 60 rational active-face optima, every accepted pair, eight full fresh refits, frozen-holdout poisoning, all 4,552 native/browser score comparisons, scalar kernels, symmetries, strict JSON errors, and real keyboard, copy, download, and mobile behaviour. Floating values are checked against stated tolerances; I don't promise bitwise portability across platforms.

Strict prediction input is `{"samples":[[30 original-unit numeric feature values]]}`. Booleans, strings, nonfinite values, duplicate keys, and unknown fields all fail.

```sh
python -m marginlab predict --input samples.json
python -m marginlab predict --input samples.json --model-id rbf-10
python -m marginlab fit --budget 0 --output unfinished.json
python -m pip wheel . --no-deps -w /tmp/wheels
python -m pip install --target /tmp/installed /tmp/wheels/marginlab-0.3.0-py3-none-any.whl
bash examples/verify_wheel.sh /tmp/installed "$PWD"
```

The wheel check imports all eleven modules, package root included, and exercises all nine main commands plus the box exporter outside the checkout, eight fresh fits and saved/fresh full replay audits included. The browser exporter rebuilds the public data and references without refitting: `python -m marginlab.export`. You can reproduce the score figure with `python examples/figure.py`.

For the full installed suite, run `python examples/check_installed.py` after installing NumPy 2.2.6, setuptools, and wheel. It builds a wheel, installs it in a fresh environment, and runs all 36 existing Python tests outside the checkout. Eleven module origins are checked before and after. Copied tests, exporters, static files, and data-only archive directories don't include the library's Python source. The suite keeps the exact oracles, fresh fits, holdout-poison control, full pair replay, public-data hashes, and figure reproduction.

The helper also compares 11 complete CLI reports covering all nine commands, through both `python -m marginlab` and the installed `marginlab` command. It compares the complete fresh eight-model report, the exact linear proof, and the exact box proof, plus their printed summaries. These source/wheel comparisons run in the same environment; they don't replace the cross-platform tolerance contract below, and they don't add independent statistical evidence. CI keeps the original wheel smoke job and adds this full installed check as a separate job.

## Where it stops

An SVM score is signed, not calibrated certainty. The first-two-feature plot holds the other 28 features at their training means: a decision slice, not a projection. Points overlay only two coordinates. An edited input can be implausible. Synthetic controls aren't causal ablations, the source labels are historical targets, and a few errors across 115 examples don't establish medical reliability.

The optimizer picks pairs by deterministic maximum violation, not every heuristic from Platt's original SMO. Dense Gram storage is quadratic, so there's no large-scale speed claim here. `budget_exhausted` and `no_progress` stay distinct from numerical tolerance. The formulas, source attribution, C sum-loss convention, exact source hashes, and the original UCI license are all in [METHOD.md](METHOD.md).

## How the artifacts are stored

The complete report is assembled from `marginlab/artifacts/report.json`, eight readable per-candidate JSON files, and three bounded linear-10 pair chunks. `saved()` restores all original values and pair order. Every stored text file stays below 600 KB, so a browser editor won't truncate it. A fresh `fit --output` still writes one complete ordinary JSON report. The source names text is stored as a JSON string, so its exact original bytes (including its missing final newline) survive text-editor normalization. Both original source hashes are verified after decoding. This is storage only: no refit, no rounding, no change of protocol.

## Reproducing it elsewhere

Floating-point reductions can change near-tied pair choices: the published artifact has 275 updates for linear C=0.01, while a Python 3.10 CI run took 267. Reproduction means the same protocol, the same selected configuration, the same stop status, all the same train/validation/test confusion counts, overlapping numerical primal/dual objective intervals, and full pair replay. It does not mean identical update counts or dual coefficients; those needn't be unique. The full-data intervals are still numerical, with a comparison slack of 5e-7 times (1 + maximum absolute primal value), not exact bounds. The holdout-poison test compares two fresh fits in the same runtime and requires exactly identical fitted models, pair histories, and validation metrics. Independent metric recomputation allows 1e-12 times (1 + absolute hinge) for the mean hinge reduction; confusion counts stay exact. None of these test corrections change any saved model, score, stopping tolerance, or data split.

## Exact linear proofs, separate from the numerical fits

Four self-contained exact rational weak-duality proofs cover the saved linear models. The original binary64 dual coefficients carry a small exact signed residual, so clipping plus a recorded coefficient reduction makes a feasible dual first. Then exact feasible primal and dual witnesses bound the represented standardized training optimum. Two separate readers, Python Fraction and browser BigInt, recompute every constraint and objective without fitting. The [proof derivation](CERTIFICATE.md) explains the scope, the exact repair, the independent readers, and the source-binding control. RBF, exact real-number standardization, generalization, authentication, and clinical claims are explicitly excluded. The original models, scores, and train/validation/test split are unchanged.

```sh
python -m marginlab certificate --model-id linear-1 --output proof.json
python -m marginlab verify-certificate --input proof.json
python examples/certificate_export.py
```

## Exact input boxes, separate from clinical reliability

The [finite original-unit boxes](https://midya8.github.io/svm-from-scratch/box.html) report exact affine minimum and maximum, attaining corners, tie-aware reachable labels, and source consistency from every saved coefficient and represented training input. The spread presets are synthetic multiples of training scales, not measurement tolerances, not plausible interventions. The certificate is about a declared rational affine function, not production floating-point execution. All 30 intervals are editable; stale or invalid results refuse export. The checks include 120 independent exhaustive-corner cases, 256 direct exact dual endpoint scores, and 128 full native/browser responses. Existing fits, scores, data, and split are untouched. [BOX.md](BOX.md) derives the established support-function calculation and documents the limits.

```sh
python -m marginlab box --input examples/box.json --output /tmp/input-box.json
python -m marginlab verify-box --input /tmp/input-box.json
python -m marginlab.box_export
```

No RBF, clinical, causal, domain-validity, generalization, authentication, or floating-inference safety guarantee.
