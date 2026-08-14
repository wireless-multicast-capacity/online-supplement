# Online Supplement

This repository accompanies the manuscript *On the Network Coding Capacity of Wireless Multicast Networks under One-Hop Interference* by Xiaohong Cai and Raymond W. Yeung.

## Contents

- `supplement.pdf`: the online supplement submitted with the manuscript.
- `supplement.tex`: LaTeX source of the supplement.
- `main-paper-labels.aux`: the six external labels from the manuscript that are required to compile `supplement.tex` independently.
- `verify_butterfly_capacity.py`: exhaustive matching-based linear-program verification for the reduced butterfly-like networks.
- `requirements.txt`: pinned Python dependencies used for the archived computation.
- `results/verification_results.jsonl`: one machine-readable record for every reduced configuration.
- `results/verification_summary.json`: aggregate counts and the final mismatch count.

## Computational verification

The program examines all

- \(2^4 3^4=1296\) reduced configurations for \(I=D\), and
- \(2^4 3^5=3888\) reduced configurations for \(I\ne D\),

for a total of 5184 configurations. For each configuration, it constructs the directed hypergraph, enumerates all matchings (including the empty matching), and solves the min-cut linear program described in Section I of the supplement. Each edge on an inner subpath is constrained to carry resource of size \(1/3\).

The archived exhaustive run has the following summary:

```text
                         I = D    I != D    Total
Configurations           1296       3888     5184
Conditions satisfied     1072       3866     4938
Rate 5/6 achieved        1072       3866     4938
Theory agrees with LP    1296       3888     5184
Nonoptimal LPs              0          0        0
Mismatches                  0          0        0
```

Here, `matching_cases` counts configurations for which the structural conditions and the LP test agree on whether rate \(5/6\) is achievable under the inner-subpath constraints.

## Reproducing the computation

Python 3.12 was used for the archived run. Install the dependencies and start the verification from the repository root:

```bash
python -m pip install -r requirements.txt
python verify_butterfly_capacity.py --workers 4
```

The default worker count is at most four. Each worker invokes one single-threaded CBC linear-program solver, so a smaller value may be preferable on machines with limited memory:

```bash
python verify_butterfly_capacity.py --workers 2
```

Progress is printed during the run. The JSONL result file is flushed after every completed configuration. If a run is interrupted, resume it with

```bash
python verify_butterfly_capacity.py --workers 4 --resume
```

For a short installation test, run

```bash
python verify_butterfly_capacity.py --workers 2 --limit-per-variant 2 --output-dir smoke-test-results
```

## Compiling the supplement

The PDF can be rebuilt with a LaTeX installation containing `amsmath`, `amssymb`, `amsthm`, and `xr`:

```bash
latexmk -pdf supplement.tex
```

The supplied `main-paper-labels.aux` contains only the external section and result numbers referenced by the supplement. It should be updated if those numbers change in a later manuscript version.
