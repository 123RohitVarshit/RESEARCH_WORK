"""
Topology Module - Genotype Definition

This module defines the "DNA" of teaching strategies:
- Action: Enum-like class of pedagogical primitives
- Topology: A sequence of actions representing a teaching strategy

The Genotype/Phenotype separation is the core innovation of EPT:
- Genotype (this module): The STRUCTURE of the teaching approach
- Phenotype (topology_classroom.py): The LLM EXECUTION of that structure

This allows us to evolve structure independently of text generation,
avoiding mode collapse issues common in standard RLHF.
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, ClassVar


class Action:
    """
    Pedagogical action primitives.
    
    These represent the fundamental moves a tutor can make.
    Each action maps to a specific instruction for the LLM.
    
    Attributes:
        DIAGNOSE: Assess student understanding without explaining
        SCAFFOLD: Break down problem into simpler steps
        HINT: Provide conceptual guidance without numbers
        VERIFY: Ask student to check their work
        ENCOURAGE: Provide positive reinforcement
    
    Example:
        >>> Action.all()
        ['diagnose', 'scaffold', 'hint', 'verify', 'encourage']
    """
    
    DIAGNOSE: ClassVar[str] = "diagnose"
    SCAFFOLD: ClassVar[str] = "scaffold"
    HINT: ClassVar[str] = "hint"
    VERIFY: ClassVar[str] = "verify"
    ENCOURAGE: ClassVar[str] = "encourage"
    
    # Instruction mappings for each action
    _INSTRUCTIONS: ClassVar[Dict[str, str]] = {
        "diagnose": (
            "Do NOT explain or solve. Ask the student what they think "
            "the first step is to approach this problem."
        ),
        "scaffold": (
            "Break the problem down into smaller pieces. Create a similar, "
            "simpler example with different numbers to illustrate the concept."
        ),
        "hint": (
            "Give a conceptual hint about the mathematical operation or "
            "formula needed, but do NOT mention any specific numbers from the problem."
        ),
        "verify": (
            "Ask the student to double-check and verify their last "
            "calculation or reasoning step."
        ),
        "encourage": (
            "Acknowledge the student's effort positively and encourage them "
            "to try the next step. Do not give the answer."
        ),
    }
    
    @classmethod
    def all(cls) -> List[str]:
        """Return all available action types."""
        return [cls.DIAGNOSE, cls.SCAFFOLD, cls.HINT, cls.VERIFY, cls.ENCOURAGE]
    
    @classmethod
    def get_instruction(cls, action: str) -> str:
        """
        Get the LLM instruction for a given action.
        
        Args:
            action: One of the action constants
            
        Returns:
            Natural language instruction for the LLM
        """
        return cls._INSTRUCTIONS.get(action, "Guide the student gently.")


@dataclass
class Topology:
    """
    A teaching strategy represented as a sequence of actions.
    
    This is the "genotype" - the DNA that defines HOW a tutor should
    approach teaching. The actual text generation (phenotype) is handled
    by TopologyConversation.
    
    Attributes:
        genes: List of action strings (e.g., ['diagnose', 'verify', 'hint'])
        fitness: Fitness score from evaluation (-999 = unevaluated)
    
    Example:
        >>> t = Topology.random_init(length=4)
        >>> print(t.genes)
        ['verify', 'diagnose', 'encourage', 'hint']
        >>> t.mutate()  # Randomly change one gene
        >>> print(t.get_instruction(0))  # Get instruction for turn 0
        "Ask the student to double-check..."
    """
    
    genes: List[str]
    fitness: float = -999.0
    
    def get_instruction(self, turn_idx: int) -> str:
        """
        Get the LLM instruction for a specific turn.
        
        Args:
            turn_idx: The current turn number (0-indexed)
            
        Returns:
            Natural language instruction for the LLM
        """
        if turn_idx >= len(self.genes):
            return "Guide the student gently toward the solution."
        
        action = self.genes[turn_idx]
        return Action.get_instruction(action)
    
    def mutate(self) -> None:
        """
        Apply single-point mutation.
        
        Randomly selects one position in the gene sequence and
        replaces it with a randomly chosen action.
        
        This is an in-place operation modifying self.genes.
        """
        if not self.genes:
            return
        
        mutation_point = random.randint(0, len(self.genes) - 1)
        new_action = random.choice(Action.all())
        self.genes[mutation_point] = new_action
    
    @classmethod
    def crossover(cls, parent1: "Topology", parent2: "Topology") -> "Topology":
        """
        Create a child topology by recombining two parents.
        
        Uses single-point crossover: a random split point is chosen,
        genes before the point come from parent1, genes after from parent2.
        
        Args:
            parent1: First parent topology
            parent2: Second parent topology
            
        Returns:
            New Topology child
        
        Example:
            >>> p1 = Topology(genes=['A', 'B', 'C', 'D'])
            >>> p2 = Topology(genes=['W', 'X', 'Y', 'Z'])
            >>> child = Topology.crossover(p1, p2)
            >>> # child.genes might be ['A', 'B', 'Y', 'Z']
        """
        # Handle edge cases
        if len(parent1.genes) < 2:
            return cls(genes=parent1.genes.copy())
        
        # Choose random crossover point (not at edges)
        split_point = random.randint(1, len(parent1.genes) - 1)
        
        # Combine genes from both parents
        child_genes = parent1.genes[:split_point] + parent2.genes[split_point:]
        
        return cls(genes=child_genes)
    
    @classmethod
    def random_init(cls, length: int = 4) -> "Topology":
        """
        Create a topology with random genes.
        
        Args:
            length: Number of genes (teaching turns)
            
        Returns:
            New Topology with random action sequence
        
        Example:
            >>> t = Topology.random_init(length=5)
            >>> len(t.genes)
            5
        """
        genes = random.choices(Action.all(), k=length)
        return cls(genes=genes)
    
    def copy(self) -> "Topology":
        """Create a deep copy of this topology."""
        return Topology(genes=self.genes.copy(), fitness=self.fitness)
    
    def __str__(self) -> str:
        """String representation showing genes and fitness."""
        return f"Topology({self.genes}, fitness={self.fitness:.1f})"
    
    def __repr__(self) -> str:
        return self.__str__()


# =============================================================================
# Module-level convenience functions
# =============================================================================

def create_random_population(size: int, gene_length: int = 4) -> List[Topology]:
    """
    Create a population of random topologies.
    
    Args:
        size: Number of individuals in population
        gene_length: Number of genes per topology
        
    Returns:
        List of random Topology objects
    """
    return [Topology.random_init(length=gene_length) for _ in range(size)]


if __name__ == "__main__":
    # Demo usage
    print("=== Topology Module Demo ===\n")
    
    # Create a random topology
    t = Topology.random_init(length=4)
    print(f"Random topology: {t.genes}")
    
    # Show instructions for each turn
    print("\nInstructions per turn:")
    for i, gene in enumerate(t.genes):
        print(f"  Turn {i} ({gene}): {t.get_instruction(i)[:50]}...")
    
    # Demonstrate mutation
    print(f"\nBefore mutation: {t.genes}")
    t.mutate()
    print(f"After mutation:  {t.genes}")
    
    # Demonstrate crossover
    p1 = Topology(genes=['diagnose', 'diagnose', 'diagnose', 'diagnose'])
    p2 = Topology(genes=['verify', 'verify', 'verify', 'verify'])
    child = Topology.crossover(p1, p2)
    print(f"\nCrossover: {p1.genes} x {p2.genes} = {child.genes}")
