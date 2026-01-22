"""
Utility Functions Module

This module provides utility functions for the evolutionary algorithm:
- Diversity measurement
- Selection algorithms
- Population statistics
"""

import random
from typing import List, Tuple

from ept.topology import Topology


def measure_structural_diversity(population: List[Topology]) -> float:
    """
    Measure the structural diversity of a population.
    
    Diversity is calculated as the ratio of unique topologies to
    total population size. A value of 1.0 means all topologies are
    unique; 0.25 means only 25% are unique (high convergence).
    
    This metric helps detect mode collapse - if diversity drops
    too low, the population has converged to similar strategies.
    
    Args:
        population: List of Topology objects
        
    Returns:
        Diversity score between 0.0 and 1.0
        
    Example:
        >>> pop = [
        ...     Topology(genes=['a', 'b']),
        ...     Topology(genes=['a', 'b']),  # duplicate
        ...     Topology(genes=['c', 'd']),
        ...     Topology(genes=['e', 'f'])
        ... ]
        >>> measure_structural_diversity(pop)
        0.75  # 3 unique out of 4
    """
    if not population:
        return 0.0
    
    # Convert gene lists to tuples for hashing
    unique_topologies = set(
        tuple(t.genes) for t in population
    )
    
    return len(unique_topologies) / len(population)


def fitness_proportional_selection(
    population: List[Topology],
    k: int = 1
) -> List[Topology]:
    """
    Select individuals using fitness-proportional (roulette wheel) selection.
    
    Each individual's selection probability is proportional to its fitness.
    This allows fitter individuals to be selected more often while still
    giving less fit individuals a chance (maintaining diversity).
    
    Args:
        population: List of Topology objects with fitness scores
        k: Number of individuals to select
        
    Returns:
        List of k selected Topology objects (may contain duplicates)
        
    Example:
        >>> pop = [
        ...     Topology(genes=['a'], fitness=100),
        ...     Topology(genes=['b'], fitness=50),
        ...     Topology(genes=['c'], fitness=10)
        ... ]
        >>> selected = fitness_proportional_selection(pop, k=2)
        >>> # 'a' is most likely to be selected
    """
    if not population:
        return []
    
    if k <= 0:
        return []
    
    # Get fitness values
    fitnesses = [p.fitness for p in population]
    
    # Shift to make all positive (handle negative fitness)
    min_fitness = min(fitnesses)
    adjusted = [f - min_fitness + 1.0 for f in fitnesses]
    
    # Calculate total for probability calculation
    total = sum(adjusted)
    
    # Handle edge case where all have same fitness
    if total < 0.01:
        return random.sample(population, min(k, len(population)))
    
    # Calculate selection probabilities
    weights = [a / total for a in adjusted]
    
    # Select k individuals with replacement
    return random.choices(population, weights=weights, k=k)


def tournament_selection(
    population: List[Topology],
    k: int = 1,
    tournament_size: int = 3
) -> List[Topology]:
    """
    Select individuals using tournament selection.
    
    For each selection, randomly pick tournament_size individuals
    and select the one with highest fitness. This provides more
    selection pressure than roulette wheel selection.
    
    Args:
        population: List of Topology objects with fitness scores
        k: Number of individuals to select
        tournament_size: Number of individuals per tournament
        
    Returns:
        List of k selected Topology objects
    """
    if not population:
        return []
    
    selected = []
    for _ in range(k):
        # Random tournament
        tournament = random.sample(
            population, 
            min(tournament_size, len(population))
        )
        # Winner is the one with highest fitness
        winner = max(tournament, key=lambda t: t.fitness)
        selected.append(winner)
    
    return selected


def get_population_stats(population: List[Topology]) -> dict:
    """
    Calculate statistics for a population.
    
    Args:
        population: List of Topology objects
        
    Returns:
        Dict with min, max, mean fitness and diversity
    """
    if not population:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "diversity": 0.0,
            "size": 0
        }
    
    fitnesses = [t.fitness for t in population]
    
    return {
        "min": min(fitnesses),
        "max": max(fitnesses),
        "mean": sum(fitnesses) / len(fitnesses),
        "diversity": measure_structural_diversity(population),
        "size": len(population)
    }


def get_elites(population: List[Topology], n: int = 1) -> List[Topology]:
    """
    Get the n fittest individuals from the population.
    
    Used for elitism - preserving the best solutions across generations.
    
    Args:
        population: List of Topology objects
        n: Number of elites to return
        
    Returns:
        List of n Topology objects with highest fitness
    """
    if not population:
        return []
    
    sorted_pop = sorted(population, key=lambda t: t.fitness, reverse=True)
    return [t.copy() for t in sorted_pop[:n]]


if __name__ == "__main__":
    # Demo
    print("=== Utility Functions Demo ===\n")
    
    # Create test population
    pop = [
        Topology(genes=['diagnose', 'verify'], fitness=100),
        Topology(genes=['hint', 'scaffold'], fitness=50),
        Topology(genes=['encourage', 'verify'], fitness=75),
        Topology(genes=['diagnose', 'verify'], fitness=80),  # duplicate genes
    ]
    
    print("Population:")
    for t in pop:
        print(f"  {t}")
    
    # Diversity
    div = measure_structural_diversity(pop)
    print(f"\nDiversity: {div:.2f}")
    
    # Stats
    stats = get_population_stats(pop)
    print(f"\nStats: {stats}")
    
    # Selection
    print("\nFitness-proportional selection (5 picks):")
    selected = fitness_proportional_selection(pop, k=5)
    for t in selected:
        print(f"  {t.genes} (fitness={t.fitness})")
    
    # Elites
    elites = get_elites(pop, n=2)
    print(f"\nTop 2 elites: {[e.genes for e in elites]}")
