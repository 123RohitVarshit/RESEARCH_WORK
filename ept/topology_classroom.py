"""
Topology Classroom Module - Phenotype Execution

This module provides the TopologyConversation class that wraps the base
Conversation class from pedagogicalrl and injects gene-based instructions.

The Genotype/Phenotype separation:
- Genotype (topology.py): The STRUCTURE of the teaching approach
- Phenotype (this module): The LLM EXECUTION of that structure

TopologyConversation overrides the get_conversation() method to inject
the current gene's instruction into the LLM prompt, constraining its
behavior while allowing natural language generation.

NOTE: This module uses lazy imports to allow testing without pedagogicalrl.
The actual TopologyConversation class is created dynamically when first needed.
"""

from typing import List, Dict, Any, Optional
from jinja2 import Template

from ept.topology import Topology


# Jinja2 template for constructing the teacher prompt
TEACHER_PROMPT_TEMPLATE = Template("""
SYSTEM: You are a Socratic Math Tutor. Your goal is to help students learn by guiding them to discover solutions themselves.

CRITICAL RULES:
1. Do NOT reveal the final answer directly
2. Do NOT solve the problem for the student
3. Guide the student through questions and hints

CURRENT STRATEGY: {{ instruction }}

PROBLEM: {{ problem }}

CONVERSATION HISTORY:
{{ history }}

YOUR RESPONSE:
""".strip())


# Cache for the dynamically created class
_TopologyConversation = None


def _get_conversation_state():
    """Get ConversationState from pedagogicalrl."""
    try:
        from pedagogicalrl.src.classroom import ConversationState
        return ConversationState
    except ImportError:
        import sys
        import os
        repo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pedagogicalrl')
        if os.path.exists(repo_path) and repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        from src.classroom import ConversationState
        return ConversationState


def _create_topology_conversation_class():
    """
    Dynamically create the TopologyConversation class.
    
    This is done lazily to avoid import errors when pedagogicalrl
    is not available (e.g., during testing or initial setup).
    """
    global _TopologyConversation
    
    if _TopologyConversation is not None:
        return _TopologyConversation
    
    # Import base class
    try:
        from pedagogicalrl.src.classroom import Conversation, ConversationState
    except ImportError:
        import sys
        import os
        repo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pedagogicalrl')
        if os.path.exists(repo_path) and repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        from src.classroom import Conversation, ConversationState
    
    class TopologyConversation(Conversation):
        """
        A Conversation wrapper that injects gene-based instructions.
        
        This is the "phenotype" - it takes the structural DNA (Topology)
        and executes it via the LLM. Each turn, it looks up the current
        gene and injects the corresponding instruction into the prompt.
        
        Attributes:
            topology: The Topology object defining the teaching strategy
            turn_count: Current turn number (for gene lookup)
            template: Jinja2 template for prompt construction
        """
        
        def __init__(
            self,
            problem: str,
            answer: str,
            generation_cfg: Any,
            topology: Topology,
            template: Optional[Template] = None
        ) -> None:
            """
            Initialize a TopologyConversation.
            
            Args:
                problem: The math problem text
                answer: The correct answer (for fitness evaluation)
                generation_cfg: Hydra configuration for generation
                topology: The Topology defining the teaching strategy
                template: Optional custom Jinja2 template
            """
            super().__init__(problem, answer, generation_cfg)
            
            self.topology = topology
            self.turn_count = 0
            self.template = template or TEACHER_PROMPT_TEMPLATE
        
        def get_conversation(self) -> List[Dict[str, str]]:
            """
            Override base method to inject gene-based instruction.
            
            When it's the teacher's turn, this method:
            1. Gets the instruction for the current gene
            2. Formats the recent conversation history
            3. Renders the prompt template
            4. Returns the formatted prompt as a chat message
            
            For student turns, it delegates to the parent class.
            
            Returns:
                List of message dicts with 'role' and 'content' keys
            """
            if self.state == ConversationState.TEACHER_TURN:
                # Get instruction for current turn's gene
                instruction = self.topology.get_instruction(self.turn_count)
                
                # Format recent conversation history (last 6 messages)
                history_messages = self.conversation[-6:] if self.conversation else []
                history = "\n".join([
                    f"{msg['role'].upper()}: {msg['content']}"
                    for msg in history_messages
                ])
                
                if not history:
                    history = "(No previous messages)"
                
                # Render the prompt template
                prompt = self.template.render(
                    problem=self.problem,
                    instruction=instruction,
                    history=history
                )
                
                # Increment turn counter
                self.turn_count += 1
                
                # Return as chat message format
                return [{"role": "user", "content": prompt}]
            
            # For other states (student turn, etc.), use parent behavior
            return super().get_conversation()
        
        def get_topology_info(self) -> Dict[str, Any]:
            """
            Get information about the current topology state.
            
            Useful for debugging and logging.
            
            Returns:
                Dict with topology genes, current turn, and state
            """
            return {
                "genes": self.topology.genes,
                "current_turn": self.turn_count,
                "state": str(self.state),
                "conversation_length": len(self.conversation)
            }
    
    _TopologyConversation = TopologyConversation
    return _TopologyConversation


def TopologyConversation(
    problem: str,
    answer: str,
    generation_cfg: Any,
    topology: Topology,
    template: Optional[Template] = None
):
    """
    Factory function that creates a TopologyConversation instance.
    
    This is a drop-in replacement for the class constructor.
    It lazily creates the class on first use.
    
    Args:
        problem: The math problem text
        answer: The correct answer (for fitness evaluation)
        generation_cfg: Hydra configuration for generation
        topology: The Topology defining the teaching strategy
        template: Optional custom Jinja2 template
        
    Returns:
        TopologyConversation instance
    """
    cls = _create_topology_conversation_class()
    return cls(problem, answer, generation_cfg, topology, template)


def create_topology_conversation(
    problem: str,
    answer: str,
    generation_cfg: Any,
    topology: Optional[Topology] = None,
    gene_length: int = 4
):
    """
    Convenience factory function for creating TopologyConversation.
    
    Args:
        problem: The math problem text
        answer: The correct answer
        generation_cfg: Hydra configuration for generation
        topology: Optional pre-created topology (random if None)
        gene_length: Length of random topology if creating new one
        
    Returns:
        Configured TopologyConversation instance
    """
    if topology is None:
        topology = Topology.random_init(length=gene_length)
    
    return TopologyConversation(
        problem=problem,
        answer=answer,
        generation_cfg=generation_cfg,
        topology=topology
    )
