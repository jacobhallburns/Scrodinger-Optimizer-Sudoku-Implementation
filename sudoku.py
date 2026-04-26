"""
File: sudoku_SRA.py
Author: Jacob Hall-Burns
Description: Scrodinger Optimizer (SRA) applied to Sudoku as a combinatorial metaheuristic,
based directly on:
Hussein et al. (2025) "Schrödinger optimizer: A quantum
             duality-driven metaheuristic for stochastic optimization and
             engineering challenges." Knowledge-Based Systems 328, 114273.
SRA to Sudoku Concept Mapping
Position x_i - Real-valued vector -> Sudoku Board (81 char list)
Wave function Psai(x_i, t) - sin(x_i) -> sin(conflict_count + e)
Fitness - Objective Value -> Conflict_count
p[i] -> ((N-i)/n)^2, ranked by fitness
TF(t) -> (t/t_max)^3
Branch 1 rand < 0.03 -> Population_initialization
Branch 2 p[i] < TF(t) -> Particle-like (newton) - Exploitation
    rand < 0.5 - 2x_i - x_{i-1} -> _swap_within_box
    rand >= 5.0 - x_best - rand*(r1-r2) -> _move_value_to_index
Branch 3 p[i] >= TF(t) - Wave-like(schrodinger) - Exploration
    rand < 0.5 - x_best + wave terms -> _move_value_to_index
    rand >= 0.5 - x_i + wave terms -> _swap_within_box
"""

import random
import math
import sys

class Particle:
    # Initiates each Particle
    def __init__(self, board_list, fixed_indices):
        self.board = board_list
        self.fixed_indices = set(fixed_indices)
        self.conflict_count = 999
        self.conflict_map = [0] * 81
        self.Psai = [0.0] * 81

    # Evaluates the Sudoku boards
    def evaluate(self):
        # Creates a new, empty conflict map
        self.conflict_map = [0] * 81
        # Checks for conflicts in each row and column
        for i in range(9):
            self._mark_conflicts([i*9 + j for j in range(9)])
            self._mark_conflicts([j*9 + i for j in range(9)])
        # Adds up the conflicts stored in the conflict map
        self.conflict_count = sum(1 for x in self.conflict_map if x > 0)
        
        # Wave function refresh (Algorithm 1: "Update the value of Psi by Eq.19")
        for j in range(81):
            self.Psai[j] = math.sin(self.conflict_map[j] + 1e-6)

        return self.conflict_count

    # Marks the conflicts found in the conflict map
    def _mark_conflicts(self, indices):
        # Creates set of seen numbers for each given column and row
        seen = {}
        # For each index in the row, we look at the number seen at that index and check to see if it has been seen before.
        # If the number has already been seen in this column or row we increase the heat of the conflict_map at that index and increase the heat of that seen value
        # If it has not been seen yet, it is added to the set of seen values
        for idx in indices:
            val = self.board[idx]
            if val in seen:
                self.conflict_map[idx] += 1
                self.conflict_map[seen[val]] += 1
            else:
                seen[val] = idx

    # Gets the indexes for the rest of the cells in the subgrid that the given index is in
    def _get_subgrid_indices_by_box(self, box_idx):
        start_row, start_col = (box_idx // 3) * 3, (box_idx % 3) * 3
        return [r * 9 + c 
                        for r in range(start_row, start_row + 3)
                        for c in range(start_col, start_col + 3)]
    
    def _box_of(self, cell_idx):
        row, col = divmod(cell_idx,9)
        return (row//3)*3+(col//3)

    # A swap between two cells within a subgrid is made randomly for the cells not in the list of given/ set indicies.
    def _swap_within_box(self, box_idx):
        indices = self._get_subgrid_indices_by_box(box_idx)
        swappable = [i for i in indices if i not in self.fixed_indices]
        if len(swappable) >= 2:
            idx1, idx2 = random.sample(swappable, 2)
            self.board[idx1], self.board[idx2] = self.board[idx2], self.board[idx1]

    # Used in Contraction, this swaps the value in the target cell towards the value of the best board it is being compared to
    def _move_value_to_index(self, target_idx, val):
        if target_idx in self.fixed_indices:
            return
        box_idx = self._box_of(target_idx)
        indices = self._get_subgrid_indices_by_box(box_idx)
        for i in indices:
            if self.board[i] == val and i not in self.fixed_indices:
                self.board[target_idx], self.board[i] = \
                    self.board[i], self.board[target_idx]
                break

# This validates the initial_board and budget value to make sure they are valid inputs for the program
def initial_validation(initial_board, budget):
    if len(initial_board) != 81 or set(initial_board) - set("123456789."):
        raise ValueError("Invalid board string.")
    if budget <= 0:
        raise ValueError("Budget must be positive.")

# Initiates a star/ Sudoku puzzle's baord state based on the initial string given (with "."'s in it for blank spots), in a subgrid valid way.
def population_initialization(initial_board):
    pop = list(initial_board)
    for box in range(9):
        indices = [r * 9 + c for r in range((box // 3) * 3, (box // 3) * 3 + 3)
                            for c in range((box % 3) * 3, (box % 3) * 3 + 3)]
        fixed = [pop[i] for i in indices if pop[i] != "."]
        missing = list(set("123456789") - set(fixed))
        random.shuffle(missing)
        for i in indices:
            if pop[i] == ".": pop[i] = missing.pop()
    return pop

def rank_probabilities(population):
    """
    p[i] = ((N - i) / N)^2
    i is the rank index (0 = best / fewest conflicts, N-1 = worst).
    Returns a list aligned with `population` (not the sorted order).
    """
    n = len(population)
    order = sorted(range(n), key=lambda k: population[k].conflict_count)
    p = [0.0] * n
    for rank, agent_idx in enumerate(order):
        p[agent_idx] = ((n - rank) / n) ** 2
    return p

def threshold_function(t, t_max):
    """
    TF(t) = (t / t_max)^3
    Starts near 0 (almost everyone uses wave-like exploration), grows to 1
    (almost everyone uses particle-like exploitation).
    """
    return (t / max(t_max, 1)) ** 3

def psai_weighted_sample(particle, candidate_cells, n_pick):
    """
    Sample n_pick cells from candidate_cells with probability proportional
    to |Psai[j]|.  Higher conflict -> higher wave amplitude -> more likely
    to be targeted.
    """
    if not candidate_cells:
        return []
    weights = [abs(particle.Psai[j]) + 1e-9 for j in candidate_cells]
    total = sum(weights)
    weights = [w / total for w in weights]
    return random.choices(candidate_cells, weights=weights, k=n_pick)

def branch_random_restart(initial_board, fixed_cells):
    # Branch 1 (rand < 0.03) - Full Random Restart
    return Particle(population_initialization(initial_board), fixed_cells)

def branch_particle_like(particle, best_particle, fixed_cells):
    """
    Branch 2 (p[i] < TF(t)) - exploitation.
    rand < 0.5  -> _swap_within_box on a conflicted box (local perturbation)
    rand >= 0.5 -> _move_value_to_index: pull a conflict cell toward X_best
    """
    candidate = Particle(list(particle.board), fixed_cells)
    candidate.conflict_map = list(particle.conflict_map)
    candidate.conflict_count = particle.conflict_count
    candidate.Psai = list(particle.Psai)

    if random.random() < 0.5:
        # momentum-based local perturbation
        conflicted_boxes = list({
            candidate._box_of(j)
            for j in range(81)
            if candidate.conflict_map[j] > 0 and j not in candidate.fixed_indices
        })
        if conflicted_boxes:
            candidate._swap_within_box(random.choice(conflicted_boxes))
        else:
            candidate._swap_within_box(random.randint(0, 8))
    else:
        # best-guided pull
        conflicted = [j for j in range(81)
                      if candidate.conflict_map[j] > 0
                      and j not in candidate.fixed_indices]
        if conflicted:
            target = random.choice(conflicted)
            desired_val = best_particle.board[target]
            candidate._move_value_to_index(target, desired_val)

    return candidate


def branch_wave_like(particle, best_particle, p_i, fixed_cells):
    """
    Branch 3 (p[i] >= TF(t)) - exploration.
    rand < 0.5  -> _move_value_to_index on Psai-weighted conflict cells
    rand >= 0.5 -> _swap_within_box on Psai-weighted conflicted boxes
    """
    candidate = Particle(list(particle.board), fixed_cells)
    candidate.conflict_map = list(particle.conflict_map)
    candidate.conflict_count = particle.conflict_count
    candidate.Psai = list(particle.Psai)

    conflicted = [j for j in range(81)
                  if candidate.conflict_map[j] > 0
                  and j not in candidate.fixed_indices]

    if random.random() < 0.5:
        # best-guided wave update
        if conflicted:
            n_target = max(1, int((1.0 - p_i) * len(conflicted)))
            targets = psai_weighted_sample(candidate, conflicted, n_target)
            for t in targets:
                candidate._move_value_to_index(t, best_particle.board[t])
    else:
        # self-centred wave update
        if conflicted:
            conflicted_boxes = list({candidate._box_of(j) for j in conflicted})
            n_boxes = max(1, int((1.0 - p_i) * len(conflicted_boxes)))
            # Weight boxes by their maximum wave amplitude
            box_weights = []
            for b in conflicted_boxes:
                cells = candidate._get_subgrid_indices_by_box(b)
                box_weights.append(
                    max(abs(candidate.Psai[c]) for c in cells) + 1e-9
                )
            total_w = sum(box_weights)
            box_weights = [w / total_w for w in box_weights]
            chosen_boxes = random.choices(
                conflicted_boxes, weights=box_weights, k=n_boxes
            )
            for b in chosen_boxes:
                candidate._swap_within_box(b)

    return candidate

def main():
    # Evaluates the input for being valid, and throws an error if it is not.
    # Creates a list of the fixed indices
    try:
        initial_board = sys.stdin.readline().strip()
        budget = int(sys.stdin.readline().strip())
        initial_validation(initial_board, budget)
        fixed_cells = [k for k, val in enumerate(initial_board) if val != "."]
    except (ValueError, EOFError) as e:
        print(f"Input error: {e}", file=sys.stderr)
        sys.exit(1)

    # Default population size is set to 50. This is arbitrary
    # The Particles of stars is initialized
    population_size = 50
    t_max = budget // population_size
    particles = [Particle(population_initialization(initial_board), fixed_cells) 
                     for _ in range(population_size)]
    
    # Evaluations are initialized 
    evaluations = 0

    # First check through the randomly subgrid valid Sudoku boards to see if a "perfect" board was found on initilization
    for particle in particles:
        particle.evaluate()
        evaluations += 1
        if particle.conflict_count == 0:
            print(f"{''.join(particle.board)}")
            print(f"Solutions Explored: {evaluations}")
            print("Conflicts: 0")
            return
    
    p = rank_probabilities(particles)
    X_best = min(particles, key=lambda s: s.conflict_count)

    global_best = Particle(list(X_best.board), fixed_cells)
    global_best.conflict_map = list(X_best.conflict_map)
    global_best.Psai = list(X_best.Psai)
    global_best.conflict_count = X_best.conflict_count

    t = 0
    # While the number of evaluations is less than the budget, keep going
    while evaluations < budget:

        tf = threshold_function(t, t_max)
        
        for i, particle in enumerate(particles):
            if evaluations >= budget:
                break

            p_i = p[i]

            # Three branches 
            if random.random() < 0.03:
                # Branch 1
                candidate = branch_random_restart(initial_board, fixed_cells)
            elif p_i < tf:
                # Branch 2
                candidate = branch_particle_like(particle, X_best, fixed_cells)
            else:
                # Branch 3
                candidate = branch_wave_like(particle, X_best, p_i, fixed_cells)

            candidate.evaluate()
            evaluations += 1

            if candidate.conflict_count == 0:
                print(f"{''.join(candidate.board)}")
                print(f"Solutions Explored: {evaluations}")
                print("Conflicts: 0")
                return

            # Greedy acceptance
            if candidate.conflict_count <= particle.conflict_count:
                particles[i] = candidate

        p = rank_probabilities(particles)
        X_best = min(particles, key=lambda s: s.conflict_count)

        if X_best.conflict_count < global_best.conflict_count:
            global_best = Particle(list(X_best.board), fixed_cells)
            global_best.conflict_map = list(X_best.conflict_map)
            global_best.Psai = list(X_best.Psai)
            global_best.conflict_count = X_best.conflict_count

        t += 1
            
    # If we do not reach zero conflicts and the number of evaluations has reached the budget, print the current board and the number of conflicts
    print(f"{''.join(global_best.board)}")
    print(f"Solutions Explored: {evaluations}")
    print(f"Conflicts: {global_best.conflict_count}")
    

if __name__ == "__main__":
    main()