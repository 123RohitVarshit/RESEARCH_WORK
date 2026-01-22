"""
Standalone EPT Classroom

A self-contained classroom implementation for API-based inference.
This removes all pedagogicalrl dependencies and uses OpenRouter directly.

Key components:
- ConversationState: Enum for conversation state machine
- Conversation: Simple state machine for teacher-student dialogue  
- Classroom: Orchestrator that calls OpenRouter API
"""

import os
import httpx
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


class ConversationState(Enum):
    """State machine for conversation flow."""
    START = 0
    TEACHER_TURN = 1
    STUDENT_TURN = 2
    END = 3


@dataclass
class Conversation:
    """
    A simple conversation container.
    
    This replaces pedagogicalrl's complex Conversation class with
    a minimal implementation focused on API-based inference.
    """
    problem: str
    answer: str
    conversation: List[Dict[str, str]] = field(default_factory=list)
    state: ConversationState = ConversationState.START
    
    # Topology integration (for EPT)
    topology: Any = None
    turn_count: int = 0
    
    def start_conversation(self) -> None:
        """Initialize the conversation."""
        self.state = ConversationState.TEACHER_TURN
        self.conversation = []
        self.turn_count = 0
    
    def add_teacher_message(self, content: str) -> None:
        """Add a teacher message and switch to student turn."""
        self.conversation.append({"role": "teacher", "content": content})
        self.state = ConversationState.STUDENT_TURN
    
    def add_student_message(self, content: str) -> None:
        """Add a student message and switch to teacher turn."""
        self.conversation.append({"role": "student", "content": content})
        self.turn_count += 1
        
        # Check if student found the answer
        if self.answer in content:
            self.state = ConversationState.END
        else:
            self.state = ConversationState.TEACHER_TURN
    
    def get_teacher_prompt(self) -> str:
        """Generate the prompt for the teacher LLM."""
        instruction = "Guide the student gently."
        if self.topology:
            instruction = self.topology.get_instruction(self.turn_count)
        
        history = "\n".join([
            f"{m['role'].upper()}: {m['content']}" 
            for m in self.conversation[-6:]
        ])
        
        return f"""SYSTEM: You are a Socratic Math Tutor. Your goal is to help students learn.

CRITICAL RULES:
1. Do NOT reveal the final answer directly
2. Do NOT solve the problem for the student
3. Guide through questions and hints only

CURRENT STRATEGY: {instruction}

PROBLEM: {self.problem}

CONVERSATION:
{history}

YOUR RESPONSE (as the teacher):"""
    
    def get_student_prompt(self) -> str:
        """Generate the prompt for the student LLM."""
        history = "\n".join([
            f"{m['role'].upper()}: {m['content']}" 
            for m in self.conversation[-6:]
        ])
        
        return f"""You are a math student named Alex trying to solve this problem.
You make occasional mistakes but try to follow the teacher's guidance.
If you figure out the answer, state it clearly.

PROBLEM: {self.problem}

CONVERSATION:
{history}

YOUR RESPONSE (as the student, be brief):"""


class Classroom:
    """
    Orchestrator for teacher-student conversations using OpenRouter API.
    
    This is a simplified version that only supports API-based inference.
    """
    
    def __init__(
        self,
        teacher_model: str = "meta-llama/llama-3.1-8b-instruct",
        student_model: str = "meta-llama/llama-3.1-8b-instruct",
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1"
    ):
        """
        Initialize the classroom.
        
        Args:
            teacher_model: Model name for teacher
            student_model: Model name for student
            api_key: OpenRouter API key (defaults to env var)
            base_url: API endpoint
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")
        
        self.client = httpx.Client(timeout=60.0)
    
    def _call_llm(self, prompt: str, model: str, max_tokens: int = 512) -> str:
        """Call the OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/EPT-Research",
        }
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
        
        try:
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[ERROR] API call failed: {e}")
            return "I need to think about this more..."
    
    def generate_next_teacher_utterances(self, conversations: List[Conversation]) -> None:
        """Generate teacher responses for all conversations."""
        for conv in conversations:
            if conv.state == ConversationState.TEACHER_TURN:
                prompt = conv.get_teacher_prompt()
                response = self._call_llm(prompt, self.teacher_model)
                conv.add_teacher_message(response)
    
    def generate_next_student_utterances(self, conversations: List[Conversation]) -> None:
        """Generate student responses for all conversations."""
        for conv in conversations:
            if conv.state == ConversationState.STUDENT_TURN:
                prompt = conv.get_student_prompt()
                response = self._call_llm(prompt, self.student_model)
                conv.add_student_message(response)


# Convenience function for EPT integration
def create_classroom_from_config(cfg: Any) -> Classroom:
    """
    Create a Classroom from Hydra config.
    
    Args:
        cfg: DictConfig with teacher_model, student_model settings
    
    Returns:
        Configured Classroom instance
    """
    teacher_model = getattr(cfg.teacher_model, 'model_name_or_path', 'meta-llama/llama-3.1-8b-instruct')
    student_model = getattr(cfg.student_model, 'model_name_or_path', 'meta-llama/llama-3.1-8b-instruct')
    
    return Classroom(
        teacher_model=teacher_model,
        student_model=student_model
    )


if __name__ == "__main__":
    # Quick test
    print("=== EPT Classroom Test ===\n")
    
    classroom = Classroom()
    conv = Conversation(
        problem="Solve for x: 3x + 12 = 27",
        answer="5"
    )
    conv.start_conversation()
    
    print(f"Problem: {conv.problem}")
    print(f"State: {conv.state.name}")
    
    # Simulate one turn
    if conv.state == ConversationState.TEACHER_TURN:
        classroom.generate_next_teacher_utterances([conv])
        print(f"\nTeacher: {conv.conversation[-1]['content'][:100]}...")
    
    print("\n[OK] Classroom test successful!")
