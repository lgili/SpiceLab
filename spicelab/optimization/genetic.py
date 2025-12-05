"""Genetic algorithm optimizers using DEAP.

This module provides evolutionary optimization algorithms:
- GeneticOptimizer: Single-objective genetic algorithm
- NSGA2Optimizer: Multi-objective optimization with NSGA-II
- Custom operators for circuit optimization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from .base import (
    Constraint,
    ObjectiveFunction,
    OptimizationConfig,
    OptimizationResult,
    Optimizer,
    ParameterBounds,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# Multi-Objective Results
# =============================================================================


@dataclass
class ParetoFront:
    """Pareto front from multi-objective optimization.

    Attributes:
        solutions: List of non-dominated solutions (parameters)
        objectives: Objective values for each solution
        n_objectives: Number of objectives
    """

    solutions: list[dict[str, float]]
    objectives: list[tuple[float, ...]]
    n_objectives: int

    def __len__(self) -> int:
        return len(self.solutions)

    def get_extreme(self, objective_index: int, minimize: bool = True) -> dict[str, float]:
        """Get solution with extreme value for an objective.

        Args:
            objective_index: Which objective to optimize
            minimize: True to get minimum, False for maximum

        Returns:
            Parameters of the extreme solution
        """
        if not self.solutions:
            raise ValueError("Pareto front is empty")

        if minimize:
            idx = min(range(len(self.objectives)), key=lambda i: self.objectives[i][objective_index])
        else:
            idx = max(range(len(self.objectives)), key=lambda i: self.objectives[i][objective_index])

        return self.solutions[idx]

    def get_knee_point(self) -> dict[str, float]:
        """Get the knee point of the Pareto front.

        The knee point is the solution closest to the ideal point
        (minimum of all objectives) in normalized space.

        Returns:
            Parameters of the knee point solution
        """
        if not self.solutions:
            raise ValueError("Pareto front is empty")

        if len(self.solutions) == 1:
            return self.solutions[0]

        # Find ideal and nadir points
        obj_array = np.array(self.objectives)
        ideal = obj_array.min(axis=0)
        nadir = obj_array.max(axis=0)

        # Normalize objectives
        ranges = nadir - ideal
        ranges[ranges == 0] = 1  # Avoid division by zero
        normalized = (obj_array - ideal) / ranges

        # Find point closest to origin (ideal) in normalized space
        distances = np.sqrt((normalized**2).sum(axis=1))
        knee_idx = int(np.argmin(distances))

        return self.solutions[knee_idx]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "solutions": self.solutions,
            "objectives": [list(obj) for obj in self.objectives],
            "n_objectives": self.n_objectives,
        }


@dataclass
class MultiObjectiveResult:
    """Result from multi-objective optimization.

    Attributes:
        pareto_front: The Pareto-optimal solutions
        n_generations: Number of generations run
        n_evaluations: Total objective evaluations
        history: Population statistics per generation
    """

    pareto_front: ParetoFront
    n_generations: int
    n_evaluations: int
    history: list[dict[str, float]] = field(default_factory=list)
    message: str = ""

    @property
    def success(self) -> bool:
        """Whether optimization found valid solutions."""
        return len(self.pareto_front) > 0

    def get_best_compromise(self) -> dict[str, float]:
        """Get the best compromise (knee point) solution."""
        return self.pareto_front.get_knee_point()


# =============================================================================
# Genetic Algorithm Optimizer
# =============================================================================


@dataclass
class GAConfig:
    """Configuration for genetic algorithm.

    Attributes:
        population_size: Number of individuals in population
        n_generations: Maximum number of generations
        crossover_prob: Probability of crossover (0-1)
        mutation_prob: Probability of mutation per gene (0-1)
        tournament_size: Size of tournament for selection
        elitism: Number of best individuals to preserve
        seed: Random seed for reproducibility
        verbose: Print progress information
    """

    population_size: int = 50
    n_generations: int = 100
    crossover_prob: float = 0.8
    mutation_prob: float = 0.1
    tournament_size: int = 3
    elitism: int = 2
    seed: int | None = None
    verbose: bool = False


class GeneticOptimizer(Optimizer):
    """Single-objective genetic algorithm optimizer.

    Uses DEAP library for evolutionary computation. Falls back to
    a simple implementation if DEAP is not available.

    Example:
        >>> optimizer = GeneticOptimizer(population_size=50, n_generations=100)
        >>> result = optimizer.optimize(objective, bounds)
        >>> print(f"Best: {result.parameters}")
    """

    def __init__(
        self,
        population_size: int = 50,
        n_generations: int = 100,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.1,
        tournament_size: int = 3,
        elitism: int = 2,
    ):
        """Initialize genetic optimizer.

        Args:
            population_size: Number of individuals in population
            n_generations: Maximum generations
            crossover_prob: Crossover probability
            mutation_prob: Mutation probability per gene
            tournament_size: Tournament selection size
            elitism: Number of elite individuals to preserve
        """
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.elitism = elitism

    @property
    def name(self) -> str:
        return "genetic_algorithm"

    def optimize(
        self,
        objective: ObjectiveFunction,
        bounds: list[ParameterBounds],
        constraints: list[Constraint] | None = None,
        config: OptimizationConfig | None = None,
    ) -> OptimizationResult[Any]:
        """Run genetic algorithm optimization.

        Args:
            objective: Objective function to minimize
            bounds: Parameter bounds
            constraints: Optional constraints (applied as penalties)
            config: Optimization configuration

        Returns:
            OptimizationResult with best solution found
        """
        if config is None:
            config = OptimizationConfig()

        rng = np.random.default_rng(config.seed)
        n_params = len(bounds)

        # Track evaluations and history
        n_evals = [0]
        best_value = [float("inf")]
        best_params: list[dict[str, float]] = [{}]
        history: list[tuple[dict[str, float], float]] = []

        def evaluate(individual: np.ndarray) -> float:
            """Evaluate an individual."""
            params = self._decode(individual, bounds)
            value = objective(params)
            n_evals[0] += 1

            # Apply constraint penalties
            if constraints:
                penalty = 0.0
                for constraint in constraints:
                    violation = constraint(params)
                    if violation < 0:
                        penalty += 1000 * abs(violation)
                value += penalty

            if value < best_value[0]:
                best_value[0] = value
                best_params[0] = params.copy()
                history.append((params.copy(), value))

            return value

        # Initialize population
        population = self._init_population(n_params, self.population_size, rng)
        fitness = np.array([evaluate(ind) for ind in population])

        # Main evolution loop
        for gen in range(self.n_generations):
            # Selection
            selected_idx = self._tournament_select(
                fitness, self.population_size - self.elitism, self.tournament_size, rng
            )
            selected = population[selected_idx].copy()

            # Crossover
            for i in range(0, len(selected) - 1, 2):
                if rng.random() < self.crossover_prob:
                    selected[i], selected[i + 1] = self._crossover(selected[i], selected[i + 1], rng)

            # Mutation
            for i in range(len(selected)):
                selected[i] = self._mutate(selected[i], self.mutation_prob, rng)

            # Evaluate offspring
            offspring_fitness = np.array([evaluate(ind) for ind in selected])

            # Elitism: keep best individuals
            elite_idx = np.argsort(fitness)[: self.elitism]
            elite = population[elite_idx]
            elite_fitness = fitness[elite_idx]

            # Combine elite and offspring
            population = np.vstack([elite, selected])
            fitness = np.concatenate([elite_fitness, offspring_fitness])

            if config.verbose and gen % 10 == 0:
                print(f"  Gen {gen}: best={best_value[0]:.6g}, mean={fitness.mean():.6g}")

        return OptimizationResult(
            success=True,
            value=best_value[0],
            parameters=best_params[0],
            n_iterations=self.n_generations,
            n_evaluations=n_evals[0],
            history=history,
            message=f"GA completed after {self.n_generations} generations",
            metadata={
                "population_size": self.population_size,
                "crossover_prob": self.crossover_prob,
                "mutation_prob": self.mutation_prob,
            },
        )

    def _init_population(
        self, n_params: int, pop_size: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Initialize random population in [0, 1] space."""
        return rng.random((pop_size, n_params))

    def _decode(self, individual: np.ndarray, bounds: list[ParameterBounds]) -> dict[str, float]:
        """Decode individual from [0, 1] to parameter space."""
        return {b.name: b.denormalize(individual[i]) for i, b in enumerate(bounds)}

    def _encode(self, params: dict[str, float], bounds: list[ParameterBounds]) -> np.ndarray:
        """Encode parameters to [0, 1] space."""
        return np.array([b.normalize(params[b.name]) for b in bounds])

    def _tournament_select(
        self, fitness: np.ndarray, n_select: int, tournament_size: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Tournament selection."""
        selected = []
        for _ in range(n_select):
            candidates = rng.choice(len(fitness), tournament_size, replace=False)
            winner = candidates[np.argmin(fitness[candidates])]
            selected.append(winner)
        return np.array(selected)

    def _crossover(
        self, parent1: np.ndarray, parent2: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulated binary crossover (SBX)."""
        eta = 20  # Distribution index
        child1 = parent1.copy()
        child2 = parent2.copy()

        for i in range(len(parent1)):
            if rng.random() < 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-10:
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))

                    child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                    child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])

                    # Clip to [0, 1]
                    child1[i] = np.clip(child1[i], 0, 1)
                    child2[i] = np.clip(child2[i], 0, 1)

        return child1, child2

    def _mutate(
        self, individual: np.ndarray, mutation_prob: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Polynomial mutation."""
        eta = 20  # Distribution index
        mutant = individual.copy()

        for i in range(len(individual)):
            if rng.random() < mutation_prob:
                u = rng.random()
                if u < 0.5:
                    delta = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))

                mutant[i] = individual[i] + delta
                mutant[i] = np.clip(mutant[i], 0, 1)

        return mutant


# =============================================================================
# NSGA-II Multi-Objective Optimizer
# =============================================================================


class NSGA2Optimizer:
    """NSGA-II multi-objective optimizer.

    Non-dominated Sorting Genetic Algorithm II for Pareto optimization.

    Example:
        >>> def objectives(params):
        ...     gain = compute_gain(params)
        ...     power = compute_power(params)
        ...     return (-gain, power)  # Maximize gain, minimize power
        >>>
        >>> optimizer = NSGA2Optimizer(population_size=100, n_generations=50)
        >>> result = optimizer.optimize(objectives, bounds, n_objectives=2)
        >>> print(f"Found {len(result.pareto_front)} Pareto solutions")
    """

    def __init__(
        self,
        population_size: int = 100,
        n_generations: int = 100,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
    ):
        """Initialize NSGA-II optimizer.

        Args:
            population_size: Population size (should be even)
            n_generations: Maximum generations
            crossover_prob: Crossover probability
            mutation_prob: Mutation probability per gene
        """
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

    @property
    def name(self) -> str:
        return "nsga2"

    def optimize(
        self,
        objectives: Callable[[dict[str, float]], tuple[float, ...]],
        bounds: list[ParameterBounds],
        n_objectives: int,
        constraints: list[Constraint] | None = None,
        config: OptimizationConfig | None = None,
    ) -> MultiObjectiveResult:
        """Run NSGA-II optimization.

        Args:
            objectives: Function returning tuple of objective values
            bounds: Parameter bounds
            n_objectives: Number of objectives
            constraints: Optional constraints
            config: Optimization configuration

        Returns:
            MultiObjectiveResult with Pareto front
        """
        if config is None:
            config = OptimizationConfig()

        rng = np.random.default_rng(config.seed)
        n_params = len(bounds)
        n_evals = [0]
        history: list[dict[str, float]] = []

        def evaluate(individual: np.ndarray) -> tuple[float, ...]:
            """Evaluate objectives for an individual."""
            params = {b.name: b.denormalize(individual[i]) for i, b in enumerate(bounds)}
            obj_values = objectives(params)
            n_evals[0] += 1

            # Apply constraint penalties
            if constraints:
                penalty = 0.0
                for constraint in constraints:
                    violation = constraint(params)
                    if violation < 0:
                        penalty += 1000 * abs(violation)
                obj_values = tuple(v + penalty for v in obj_values)

            return obj_values

        # Initialize population
        population = rng.random((self.population_size, n_params))
        fitness = np.array([evaluate(ind) for ind in population])

        # Main evolution loop
        for gen in range(self.n_generations):
            # Create offspring
            offspring = []
            offspring_fitness = []

            for _ in range(self.population_size // 2):
                # Select parents using binary tournament on rank
                p1_idx = self._binary_tournament(fitness, rng)
                p2_idx = self._binary_tournament(fitness, rng)

                # Crossover
                if rng.random() < self.crossover_prob:
                    c1, c2 = self._sbx_crossover(population[p1_idx], population[p2_idx], rng)
                else:
                    c1, c2 = population[p1_idx].copy(), population[p2_idx].copy()

                # Mutation
                c1 = self._polynomial_mutation(c1, self.mutation_prob, rng)
                c2 = self._polynomial_mutation(c2, self.mutation_prob, rng)

                offspring.extend([c1, c2])
                offspring_fitness.extend([evaluate(c1), evaluate(c2)])

            offspring_arr = np.array(offspring)
            offspring_fitness_arr = np.array(offspring_fitness)

            # Combine parent and offspring
            combined_pop = np.vstack([population, offspring_arr])
            combined_fitness = np.vstack([fitness, offspring_fitness_arr])

            # Non-dominated sorting and selection
            population, fitness = self._select_next_generation(
                combined_pop, combined_fitness, self.population_size
            )

            # Record history
            if gen % 10 == 0:
                fronts = self._fast_non_dominated_sort(fitness)
                history.append(
                    {
                        "generation": gen,
                        "pareto_size": len(fronts[0]),
                        "mean_obj1": float(fitness[:, 0].mean()),
                    }
                )

                if config.verbose:
                    print(f"  Gen {gen}: Pareto size={len(fronts[0])}")

        # Extract final Pareto front
        fronts = self._fast_non_dominated_sort(fitness)
        pareto_idx = fronts[0]

        pareto_solutions = [
            {b.name: b.denormalize(population[i][j]) for j, b in enumerate(bounds)}
            for i in pareto_idx
        ]
        pareto_objectives = [tuple(fitness[i]) for i in pareto_idx]

        return MultiObjectiveResult(
            pareto_front=ParetoFront(
                solutions=pareto_solutions,
                objectives=pareto_objectives,
                n_objectives=n_objectives,
            ),
            n_generations=self.n_generations,
            n_evaluations=n_evals[0],
            history=history,
            message=f"NSGA-II completed after {self.n_generations} generations",
        )

    def _fast_non_dominated_sort(self, fitness: np.ndarray) -> list[list[int]]:
        """Fast non-dominated sorting algorithm."""
        n = len(fitness)
        domination_count = np.zeros(n, dtype=int)
        dominated_by: list[list[int]] = [[] for _ in range(n)]
        fronts: list[list[int]] = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(fitness[i], fitness[j]):
                    dominated_by[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(fitness[j], fitness[i]):
                    dominated_by[j].append(i)
                    domination_count[i] += 1

            if domination_count[i] == 0:
                fronts[0].append(i)

        current_front = 0
        while current_front < len(fronts) and fronts[current_front]:
            next_front: list[int] = []
            for i in fronts[current_front]:
                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front += 1
            if next_front:
                fronts.append(next_front)

        # Remove empty fronts
        return [f for f in fronts if f]

    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """Check if obj1 dominates obj2 (all <= and at least one <)."""
        return bool(np.all(obj1 <= obj2) and np.any(obj1 < obj2))

    def _crowding_distance(self, fitness: np.ndarray, front: list[int]) -> np.ndarray:
        """Calculate crowding distance for a front."""
        n = len(front)
        if n <= 2:
            return np.full(n, np.inf)

        distances = np.zeros(n)
        n_obj = fitness.shape[1]

        for m in range(n_obj):
            sorted_idx = np.argsort(fitness[front, m])
            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf

            obj_range = fitness[front[sorted_idx[-1]], m] - fitness[front[sorted_idx[0]], m]
            if obj_range > 0:
                for i in range(1, n - 1):
                    distances[sorted_idx[i]] += (
                        fitness[front[sorted_idx[i + 1]], m]
                        - fitness[front[sorted_idx[i - 1]], m]
                    ) / obj_range

        return distances

    def _select_next_generation(
        self, population: np.ndarray, fitness: np.ndarray, target_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select next generation using non-dominated sorting and crowding distance."""
        fronts = self._fast_non_dominated_sort(fitness)

        selected: list[int] = []
        front_idx = 0

        while len(selected) + len(fronts[front_idx]) <= target_size:
            selected.extend(fronts[front_idx])
            front_idx += 1
            if front_idx >= len(fronts):
                break

        # Fill remaining slots using crowding distance
        if len(selected) < target_size and front_idx < len(fronts):
            remaining = target_size - len(selected)
            distances = self._crowding_distance(fitness, fronts[front_idx])
            sorted_by_distance = np.argsort(-distances)  # Descending
            selected.extend([fronts[front_idx][i] for i in sorted_by_distance[:remaining]])

        return population[selected], fitness[selected]

    def _binary_tournament(self, fitness: np.ndarray, rng: np.random.Generator) -> int:
        """Binary tournament selection based on dominance."""
        i, j = rng.choice(len(fitness), 2, replace=False)
        if self._dominates(fitness[i], fitness[j]):
            return int(i)
        elif self._dominates(fitness[j], fitness[i]):
            return int(j)
        else:
            return int(rng.choice([i, j]))

    def _sbx_crossover(
        self, parent1: np.ndarray, parent2: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulated binary crossover."""
        eta = 20
        child1 = parent1.copy()
        child2 = parent2.copy()

        for i in range(len(parent1)):
            if rng.random() < 0.5 and abs(parent1[i] - parent2[i]) > 1e-10:
                u = rng.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))

                child1[i] = np.clip(0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i]), 0, 1)
                child2[i] = np.clip(0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i]), 0, 1)

        return child1, child2

    def _polynomial_mutation(
        self, individual: np.ndarray, mutation_prob: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Polynomial mutation."""
        eta = 20
        mutant = individual.copy()

        for i in range(len(individual)):
            if rng.random() < mutation_prob:
                u = rng.random()
                if u < 0.5:
                    delta = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))

                mutant[i] = np.clip(individual[i] + delta, 0, 1)

        return mutant


# =============================================================================
# Factory Functions
# =============================================================================


def get_genetic_optimizer(method: str = "ga", **kwargs: Any) -> GeneticOptimizer | NSGA2Optimizer:
    """Get a genetic algorithm optimizer by name.

    Args:
        method: "ga" for single-objective, "nsga2" for multi-objective
        **kwargs: Optimizer parameters

    Returns:
        Optimizer instance
    """
    method_lower = method.lower().replace("-", "").replace("_", "")
    if method_lower in ("ga", "genetic"):
        return GeneticOptimizer(**kwargs)
    elif method_lower in ("nsga2", "nsgaii"):
        return NSGA2Optimizer(**kwargs)
    else:
        raise ValueError(f"Unknown genetic optimizer: {method}. Use 'ga' or 'nsga2'.")


__all__ = [
    "GAConfig",
    "GeneticOptimizer",
    "NSGA2Optimizer",
    "ParetoFront",
    "MultiObjectiveResult",
    "get_genetic_optimizer",
]
