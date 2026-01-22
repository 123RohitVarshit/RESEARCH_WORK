"""
EPT (Evolutionary Pedagogical Topologies) Package

A research framework for evolving teaching strategies using genetic algorithms.
This package extends the pedagogicalrl framework with evolutionary optimization.

Main Components:
    - Topology: The "DNA" of a teaching strategy (Genotype)
    - TopologyConversation: LLM wrapper that executes strategies (Phenotype)
    - Evolution functions: Selection, mutation, crossover, fitness

Example:
    from ept import Topology
    from ept.topology_classroom import TopologyConversation
    from ept.evolution import run_evolution
    
    # Create a random teaching strategy
    strategy = Topology.random_init(length=4)
    print(strategy.genes)  # e.g., ['diagnose', 'verify', 'encourage', 'hint']

Note:
    TopologyConversation requires pedagogicalrl to be installed.
    Run setup_project.py to clone and configure the base repository.
"""

from ept.topology import Action, Topology
from ept.fitness import calculate_fitness
from ept.utils import measure_structural_diversity

__version__ = "0.1.0"
__author__ = "Rohit Varshit"

# Note: TopologyConversation is not imported here to avoid
# requiring pedagogicalrl at import time. Import it explicitly:
#   from ept.topology_classroom import TopologyConversation

__all__ = [
    "Action",
    "Topology",
    "calculate_fitness",
    "measure_structural_diversity",
]
