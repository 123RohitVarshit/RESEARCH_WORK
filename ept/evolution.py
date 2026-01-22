"""
Evolution Module

This module contains the main evolutionary algorithm loop and
evaluation functions for evolving teaching topologies.

The evolution process:
1. Initialize random population
2. Evaluate each topology across multiple problems
3. Select high-fitness individuals
4. Reproduce via mutation and crossover
5. Repeat for N generations
"""

import random
import copy
import logging
from typing import List, Dict, Any, Optional, Callable
import numpy as np

from ept.topology import Topology, create_random_population
from ept.fitness import calculate_fitness
from ept.utils import (
    fitness_proportional_selection,
    get_elites,
    get_population_stats,
    measure_structural_diversity
)

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Problem Sets
# =============================================================================

DEFAULT_PROBLEMS = [
    {"problem": "Solve for x: 3x + 12 = 27", "answer": "5"},
    {"problem": "Solve for y: 2y - 8 = 10", "answer": "9"},
    {"problem": "Solve for z: 5z + 3 = 18", "answer": "3"},
]


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_topology(
    topology: Topology,
    problems: List[Dict[str, str]],
    classroom: Any,
    generation_config: Any,
    max_turns: int = 5,
    conversation_factory: Optional[Callable] = None
) -> float:
    """
    Evaluate a single topology across multiple problems.
    
    The topology is tested on each problem, simulating a multi-turn
    conversation. The final fitness is the mean across all problems.
    
    Args:
        topology: The Topology to evaluate
        problems: List of problem dicts with 'problem' and 'answer' keys
        classroom: Classroom instance for generating utterances
        generation_config: Configuration for generation
        max_turns: Maximum conversation turns
        conversation_factory: Optional function to create conversations
        
    Returns:
        Mean fitness score across all problems
    """
    # Use standalone classroom (no pedagogicalrl dependency)
    from ept.classroom import Conversation, ConversationState
    
    scores = []
    
    for problem_data in problems:
        # Create conversation for this problem
        conv = Conversation(
            problem=problem_data["problem"],
            answer=problem_data["answer"],
            topology=topology
        )
        conv.start_conversation()
        
        # Run conversation for max_turns
        for _ in range(max_turns):
            if conv.state == ConversationState.TEACHER_TURN:
                classroom.generate_next_teacher_utterances([conv])
            elif conv.state == ConversationState.STUDENT_TURN:
                classroom.generate_next_student_utterances([conv])
            elif conv.state == ConversationState.END:
                break  # Student found the answer
        
        # Calculate fitness for this conversation
        score = calculate_fitness(
            conversation=conv.conversation,
            answer=problem_data["answer"],
            max_turns=max_turns
        )
        scores.append(score)
    
    return float(np.mean(scores))


# =============================================================================
# Evolution Operators
# =============================================================================

def create_next_generation(
    population: List[Topology],
    population_size: int,
    mutation_rate: float = 0.6,
    elite_count: int = 1
) -> List[Topology]:
    """
    Create the next generation from the current population.
    
    Process:
    1. Preserve elite individuals unchanged
    2. Fill remaining slots via mutation or crossover
    
    Args:
        population: Current population with fitness scores
        population_size: Target size for new population
        mutation_rate: Probability of mutation vs crossover
        elite_count: Number of top individuals to preserve
        
    Returns:
        New population of Topology objects
    """
    # Start with elites
    new_population = get_elites(population, n=elite_count)
    
    # Fill remaining slots
    while len(new_population) < population_size:
        if random.random() < mutation_rate:
            # Mutation: select one parent and mutate
            parent = fitness_proportional_selection(population, k=1)[0]
            child = parent.copy()
            child.mutate()
            child.fitness = -999.0  # Reset fitness for re-evaluation
        else:
            # Crossover: select two parents and recombine
            parents = fitness_proportional_selection(population, k=2)
            child = Topology.crossover(parents[0], parents[1])
            child.fitness = -999.0
        
        new_population.append(child)
    
    return new_population


# =============================================================================
# Main Evolution Loop
# =============================================================================

def run_evolution(
    classroom: Any,
    generation_config: Any,
    problems: Optional[List[Dict[str, str]]] = None,
    population_size: int = 4,
    generations: int = 4,
    gene_length: int = 4,
    max_turns: int = 5,
    mutation_rate: float = 0.6,
    elite_count: int = 1,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run the complete evolutionary optimization.
    
    This is the main entry point for the evolutionary algorithm.
    It initializes a population, evaluates and evolves it over
    multiple generations, and returns the results.
    
    Args:
        classroom: Classroom instance for LLM interaction
        generation_config: Hydra configuration for generation
        problems: List of problem dicts (uses defaults if None)
        population_size: Number of individuals per generation
        generations: Number of evolutionary generations
        gene_length: Length of topology genes
        max_turns: Max conversation turns per evaluation
        mutation_rate: Probability of mutation vs crossover
        elite_count: Number of elites to preserve
        verbose: Print progress to console
        
    Returns:
        Dict with:
            - 'best_topology': Best Topology found
            - 'history': Dict of metrics over generations
            - 'final_population': Last generation's population
    """
    problems = problems or DEFAULT_PROBLEMS
    
    if verbose:
        print("\n=== EVOLUTIONARY TOPOLOGY SEARCH ===")
        print("=" * 60)
    
    # Initialize population
    population = create_random_population(population_size, gene_length)
    
    # Track metrics
    history = {
        "best_fitness": [],
        "avg_fitness": [],
        "diversity": [],
        "best_genes": []
    }
    
    best_overall: Optional[Topology] = None
    
    # Evolution loop
    for gen in range(generations):
        if verbose:
            print(f"\n>> GENERATION {gen + 1}/{generations}")
        
        # Evaluate each topology
        for i, topology in enumerate(population):
            topology.fitness = evaluate_topology(
                topology=topology,
                problems=problems,
                classroom=classroom,
                generation_config=generation_config,
                max_turns=max_turns
            )
            
            status = "[OK]" if topology.fitness > 60 else "[--]"
            if verbose:
                print(f"   [Org {i}] {topology.genes} | Score: {topology.fitness:.1f} {status}")
        
        # Calculate stats
        stats = get_population_stats(population)
        
        # Track best
        best = max(population, key=lambda t: t.fitness)
        if best_overall is None or best.fitness > best_overall.fitness:
            best_overall = best.copy()
        
        # Record history
        history["best_fitness"].append(stats["max"])
        history["avg_fitness"].append(stats["mean"])
        history["diversity"].append(stats["diversity"])
        history["best_genes"].append(best.genes.copy())
        
        if verbose:
            print(f"   → Best={stats['max']:.1f}, Avg={stats['mean']:.1f}, Div={stats['diversity']:.2f}")
        
        # Create next generation (skip on last iteration)
        if gen < generations - 1:
            population = create_next_generation(
                population=population,
                population_size=population_size,
                mutation_rate=mutation_rate,
                elite_count=elite_count
            )
    
    # Final results
    if verbose:
        print("\n" + "=" * 60)
        print("=== FINAL RESULTS ===")
        print(f"Start Best: {history['best_fitness'][0]:.1f} -> End Best: {history['best_fitness'][-1]:.1f}")
        print(f"Diversity: {history['diversity'][-1]:.2f}")
        print(f"Best Strategy: {best_overall.genes}")
        print("=" * 60)
    
    return {
        "best_topology": best_overall,
        "history": history,
        "final_population": population
    }


if __name__ == "__main__":
    print("Evolution module loaded successfully.")
    print("To run evolution, use: python run_evolution.py")
