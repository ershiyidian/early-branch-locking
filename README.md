<h2><p align="center">
Locked at the Entrance, Open Inside: Where RLVR Narrows the Solution Space
</p></h2>

<strong><p align="center">
Qiancheng Zhou<sup>♥</sup> &nbsp; Ruizhe Li<sup>♦</sup> &nbsp;
</p></strong>

<p align="center">Shanghai University ♥ &nbsp; University of Birmingham ♦ &nbsp;</p>

<p align="center"><a href="mailto:zhou_3721@shu.edu.cn">zhou_3721@shu.edu.cn</a> &nbsp; <a href="mailto:r.li.7@bham.ac.uk">r.li.7@bham.ac.uk</a> &nbsp;</p>

---

## Overview

Research software for *Early Branch Locking in Reinforcement Learning via Verifiable Rewards*.
The package measures solution coverage, entrance concentration, access/execution behavior,
and controlled interventions on Countdown and mathematical benchmarks.

## Package layout

- `src/early_branch_locking/core`: reusable parsers, solvers, protocols, and metrics.
- `src/early_branch_locking/countdown`: Countdown analyses and interventions.
- `src/early_branch_locking/math_transfer`: benchmark and prefix-transfer analyses.
- `src/early_branch_locking/train`: optional host-runtime training adapters.
- `third_party`: pinned-source and license records; external code is not vendored.
- `tests/smoke`: dependency-light installation and API checks.

## Installation

CPU analysis:

```bash
python -m pip install -e '.[dev]'
```

Inference analyses require `python -m pip install -e '.[inference]'`; training adapters
also require `.[training]` and a compatible TinyZero/veRL checkout. Run from any directory:

```bash
python -m early_branch_locking.countdown.collect_rollouts --help
python -m early_branch_locking.math_transfer.evaluate_math_benchmarks --help
```

## Reproduction boundary

Supply benchmark files, model checkpoints, and output directories explicitly. This repository
does not distribute weights, raw rollouts, private credentials, or reported result tables.
The official math evaluator is optional: pass `--evaluator-root /path/to/math_eval` to its
scoring entry point and record its upstream commit in the output manifest. Host training
adapters require external TinyZero/veRL and are not run by the default CI.
The default installation intentionally supports only dependency-light `core` imports;
GPU-backed modules are validated after installing the matching optional extra.

## License and citation

Original code is MIT licensed; upstream components retain their own licenses as documented
in `third_party/MANIFEST.yaml` and `THIRD_PARTY_NOTICES.md`.

If you use this repository, dataset, or code in your research, please cite:

```bibtex
@inproceedings{zhou2026locked,
  title     = {Locked at the Entrance, Open Inside: Where RLVR Narrows the Solution Space},
  author    = {Zhou, Qiancheng and Li, Ruizhe},
  booktitle = {arxiv},
  year      = {2026}
}
```
