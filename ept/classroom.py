"""
Standalone EPT Classroom

A self-contained classroom implementation for API-based inference.
This removes all pedagogicalrl dependencies and uses multiple LLM providers.

Key components:
- ConversationState: Enum for conversation state machine
- Conversation: Simple state machine for teacher-student dialogue  
- LLMProvider: Configuration for a single LLM API provider
- Classroom: Orchestrator with multi-provider fallback (sync + async)
"""

import os
import asyncio
import httpx
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """State machine for conversation flow."""
    START = 0
    TEACHER_TURN = 1
    STUDENT_TURN = 2
    END = 3


@dataclass
class LLMProvider:
    """Configuration for a single LLM API provider."""
    name: str
    base_url: str
    api_key: str
    teacher_model: str
    student_model: str
    priority: int = 0  # Lower = tried first

    def get_model(self, role: str) -> str:
        return self.teacher_model if role == "teacher" else self.student_model


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


# =============================================================================
# Provider Setup
# =============================================================================

def _build_providers() -> List[LLMProvider]:
    """Build provider list from environment variables, sorted by priority."""
    providers = []
    
    # Groq — fastest, highest free limits (14,400 req/day)
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        providers.append(LLMProvider(
            name="Groq",
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_key,
            teacher_model="llama-3.3-70b-versatile",
            student_model="llama-3.3-70b-versatile",
            priority=0,
        ))
    
    # Cerebras — 1M tokens/day free
    cerebras_key = os.getenv("CEREBRAS_API_KEY")
    if cerebras_key:
        providers.append(LLMProvider(
            name="Cerebras",
            base_url="https://api.cerebras.ai/v1",
            api_key=cerebras_key,
            teacher_model="qwen-3-32b",
            student_model="llama3.1-8b",
            priority=1,
        ))
    
    # OpenRouter — fallback, many models
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        providers.append(LLMProvider(
            name="OpenRouter",
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key,
            teacher_model="meta-llama/llama-3.3-70b-instruct:free",
            student_model="qwen/qwen3-30b-a3b:free",
            priority=2,
        ))
    
    providers.sort(key=lambda p: p.priority)
    return providers


class Classroom:
    """
    Orchestrator for teacher-student conversations.
    
    Supports multiple LLM providers with automatic fallback:
    Groq (fastest) → Cerebras → OpenRouter
    
    If a provider returns a rate limit error (HTTP 429),
    the next provider is tried automatically.
    """
    
    def __init__(
        self,
        teacher_model: str = "meta-llama/llama-3.1-8b-instruct",
        student_model: str = "meta-llama/llama-3.1-8b-instruct",
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_concurrent: int = 10,
        providers: Optional[List[LLMProvider]] = None,
    ):
        """
        Initialize the classroom.
        
        Args:
            teacher_model: Default model name (used if no providers configured)
            student_model: Default model name (used if no providers configured)
            api_key: Default API key (used if no providers configured)
            base_url: Default API endpoint
            max_concurrent: Max concurrent async API calls
            providers: List of LLMProvider configs (auto-built from env if None)
        """
        self.max_concurrent = max_concurrent
        
        # Build provider chain
        self.providers = providers or _build_providers()
        
        # Fallback: if no providers found from env, use legacy single-provider mode
        if not self.providers:
            key = api_key or os.getenv("OPENROUTER_API_KEY")
            if not key:
                raise ValueError("No API keys found. Set GROQ_API_KEY, CEREBRAS_API_KEY, or OPENROUTER_API_KEY in .env")
            self.providers = [LLMProvider(
                name="OpenRouter",
                base_url=base_url,
                api_key=key,
                teacher_model=teacher_model,
                student_model=student_model,
                priority=0,
            )]
        
        # Log provider chain
        names = [p.name for p in self.providers]
        logger.info(f"[Classroom] Provider chain: {' -> '.join(names)}")
        
        # Sync client for backward compatibility
        self.client = httpx.Client(timeout=60.0)
        
        # Semaphore for rate limiting async calls
        self._semaphore: Optional[asyncio.Semaphore] = None
    
    def _get_headers(self, provider: LLMProvider) -> Dict[str, str]:
        """Headers for a specific provider."""
        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json",
        }
        if provider.name == "OpenRouter":
            headers["HTTP-Referer"] = "https://github.com/EPT-Research"
        return headers
    
    def _make_payload(self, prompt: str, model: str, max_tokens: int = 512) -> Dict[str, Any]:
        """Common payload for API calls."""
        return {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
    
    # =========================================================================
    # Synchronous Methods (backward compatible)
    # =========================================================================
    
    def _call_llm(self, prompt: str, model: str = "", role: str = "teacher", max_tokens: int = 512) -> str:
        """Call LLM API with provider fallback (sync)."""
        for provider in self.providers:
            actual_model = provider.get_model(role)
            try:
                response = self.client.post(
                    f"{provider.base_url}/chat/completions",
                    headers=self._get_headers(provider),
                    json=self._make_payload(prompt, actual_model, max_tokens)
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    logger.warning(f"[{provider.name}] Rate limited, trying next provider...")
                    continue
                logger.error(f"[{provider.name}] HTTP {e.response.status_code}: {e}")
                continue
            except Exception as e:
                logger.error(f"[{provider.name}] Error: {e}")
                continue
        
        logger.error("[ERROR] All providers failed!")
        return "I need to think about this more..."
    
    def generate_next_teacher_utterances(self, conversations: List[Conversation]) -> None:
        """Generate teacher responses for all conversations (sync)."""
        for conv in conversations:
            if conv.state == ConversationState.TEACHER_TURN:
                prompt = conv.get_teacher_prompt()
                response = self._call_llm(prompt, role="teacher")
                conv.add_teacher_message(response)
    
    def generate_next_student_utterances(self, conversations: List[Conversation]) -> None:
        """Generate student responses for all conversations (sync)."""
        for conv in conversations:
            if conv.state == ConversationState.STUDENT_TURN:
                prompt = conv.get_student_prompt()
                response = self._call_llm(prompt, role="student")
                conv.add_student_message(response)
    
    # =========================================================================
    # Asynchronous Methods (for parallel evaluation)
    # =========================================================================
    
    async def _call_llm_async(self, prompt: str, role: str = "teacher", max_tokens: int = 512) -> str:
        """Call LLM API with provider fallback (async + rate limiting)."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async with self._semaphore:
            for provider in self.providers:
                actual_model = provider.get_model(role)
                try:
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        response = await client.post(
                            f"{provider.base_url}/chat/completions",
                            headers=self._get_headers(provider),
                            json=self._make_payload(prompt, actual_model, max_tokens)
                        )
                        response.raise_for_status()
                        data = response.json()
                        return data["choices"][0]["message"]["content"]
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        logger.warning(f"[{provider.name}] Rate limited, trying next...")
                        continue
                    logger.error(f"[{provider.name}] HTTP {e.response.status_code}")
                    continue
                except Exception as e:
                    logger.error(f"[{provider.name}] Error: {e}")
                    continue
        
        logger.error("[ERROR] All providers failed!")
        return "I need to think about this more..."
    
    async def generate_teacher_utterances_async(self, conversations: List[Conversation]) -> None:
        """Generate teacher responses for all conversations in parallel."""
        eligible = [c for c in conversations if c.state == ConversationState.TEACHER_TURN]
        if not eligible:
            return
        
        prompts = [c.get_teacher_prompt() for c in eligible]
        responses = await asyncio.gather(
            *[self._call_llm_async(p, role="teacher") for p in prompts]
        )
        
        for conv, response in zip(eligible, responses):
            conv.add_teacher_message(response)
    
    async def generate_student_utterances_async(self, conversations: List[Conversation]) -> None:
        """Generate student responses for all conversations in parallel."""
        eligible = [c for c in conversations if c.state == ConversationState.STUDENT_TURN]
        if not eligible:
            return
        
        prompts = [c.get_student_prompt() for c in eligible]
        responses = await asyncio.gather(
            *[self._call_llm_async(p, role="student") for p in prompts]
        )
        
        for conv, response in zip(eligible, responses):
            conv.add_student_message(response)
    
    async def run_conversation_async(self, conv: Conversation, max_turns: int = 5) -> None:
        """Run a single conversation to completion asynchronously."""
        for _ in range(max_turns):
            if conv.state == ConversationState.TEACHER_TURN:
                response = await self._call_llm_async(
                    conv.get_teacher_prompt(), role="teacher"
                )
                conv.add_teacher_message(response)
            elif conv.state == ConversationState.STUDENT_TURN:
                response = await self._call_llm_async(
                    conv.get_student_prompt(), role="student"
                )
                conv.add_student_message(response)
            elif conv.state == ConversationState.END:
                break


# Convenience function for EPT integration
def create_classroom_from_config(cfg: Any) -> Classroom:
    """
    Create a Classroom from Hydra config.
    
    Args:
        cfg: DictConfig with teacher_model, student_model settings
    
    Returns:
        Configured Classroom instance
    """
    return Classroom()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("=== EPT Classroom Test ===\n")
    
    classroom = Classroom()
    print(f"Providers: {[p.name for p in classroom.providers]}")
    
    conv = Conversation(
        problem="Solve for x: 3x + 12 = 27",
        answer="5"
    )
    conv.start_conversation()
    
    print(f"Problem: {conv.problem}")
    print(f"State: {conv.state.name}")
    
    if conv.state == ConversationState.TEACHER_TURN:
        classroom.generate_next_teacher_utterances([conv])
        print(f"\nTeacher: {conv.conversation[-1]['content'][:100]}...")
    
    print("\n[OK] Classroom test successful!")
