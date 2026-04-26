# Scrodinger-Optimizer-Sudoku-Implementation

A combinatorial metaheuristic Sudoku solver implementing the **Schrödinger Optimizer Algorithm (SRA)**, based directly on:
> Hussein et al. (2025) *"Schrödinger optimizer: A quantum duality-driven metaheuristic for stochastic optimization and engineering challenges."* Knowledge-Based Systems 328, 114273.
> 
---

## How to Run

The program reads from stdin: one line for the board and one for the evaluation budget.
```
echo ".9.2.1.....4..8.7..7..69..814...58...6.....2...86...472..34..6..3.1..7.....8.2.1.
700000" | python3 sudoku.py
```

**Board format:** 81 characters, digits `1–9` for given cells, `.` for blanks, left-to-right row by row.

**Output:**
```
896271453214538679573469128147925836362784521958613947281347965435196782679852314
Solutions Explored: 700000
Conflicts: 10
```

---

## Algorithm Overview

SRA treats each candidate Sudoku board as a **particle** with a scalar wave function `Ψ = sin(conflict_count + ε)`, inspired by Schrödinger's equation. A population of 50 particles evolves over iterations, with each particle updated by one of three branches, as specified in the paper's Algorithm 1.

### Initialisation

Every board is filled in a **subgrid-valid** way: each 3×3 box contains exactly the digits 1–9, so box constraints are never violated. Only row and column conflicts need to be resolved.

### Wave function
`Ψ(x_i, t) = sin(conflict_count + ε)`

A per-agent scalar derived from total conflict count, mirroring the paper's `sin(x_i)` (Eq. 19). Higher conflicts → higher amplitude.

### Rank probability  `p[i]`  (Eq. 22)
After each evaluation pass, agents are ranked by conflict count (ascending). Each agent receives:

```
p[i] = ((N - rank) / N)²
```

Best agents get `p[i]` near 1, worst get near 0.

### Threshold function  `TF(t)`  (Eq. 26)
```
TF(t) = (t / t_max)³
```
Starts near 0 and grows to 1, shifting the population from exploration toward exploitation over time.

### Three update branches

| Condition | Branch | SRA equations | Sudoku operator |
|---|---|---|---|
| `rand < 0.03` | Random restart | Eq. 25 | Full `population_initialization` |
| `p[i] < TF(t)` | **Particle-like** (exploitation) | Eqs. 23–24 | `_swap_within_box` or `_move_value_to_index` toward best |
| `p[i] ≥ TF(t)` | **Wave-like** (exploration) | Eqs. 20–21 | `_move_value_to_index` or `_swap_within_box` on conflicted cells |

Within branches 2 and 3, a 50/50 coin flip selects between the two sub-equations.

### Move operators
- **`_swap_within_box(box_idx)`** — swaps two non-fixed cells within a 3×3 box. Subgrid validity is always preserved.
- **`_move_value_to_index(target, val)`** — places a target value into a cell by swapping within its box, contracting toward the best board's value.

### Greedy acceptance
A candidate replaces its parent only if `conflict_count ≤ parent.conflict_count`.

---

## Parameters

| Parameter | Value | Notes |
|---|---|---|
| Population size | 50 | Arbitrary default |
| Restart probability | 0.03 | Fixed per paper |
| `t_max` | `budget // population_size` | Derived from budget |
| Evaluation budget | User-supplied via stdin | Hard stop |

---

## SRA → Sudoku Concept Mapping

| SRA (continuous) | Sudoku (discrete) |
|---|---|
| Position vector `x_i` | 81-cell board |
| `Ψ(x_i) = sin(x_i)` | `sin(conflict_count + ε)` |
| Objective value | `conflict_count` (minimise to 0) |
| `p[i] = ((N−i)/N)²` | Same, ranked by conflict count |
| `TF(t) = (t/t_max)³` | Same |
| Eq. 23 momentum move | `_swap_within_box` on conflicted box |
| Eq. 24 best-guided move | `_move_value_to_index` toward `X_best` |
| Eq. 20 wave + best | `_move_value_to_index` on conflicted cells |
| Eq. 21 wave + self | `_swap_within_box` on conflicted boxes |
| Eq. 25 random restart | `population_initialization` |

---

## Dependencies

Python 3 standard library only (`random`, `math`, `sys`).
