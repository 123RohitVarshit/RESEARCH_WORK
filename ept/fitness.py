"""
Fitness Module

This module contains functions for evaluating the fitness of teaching
strategies (Topologies) based on conversation outcomes.

Fitness evaluation considers:
1. Student success (did they arrive at the correct answer?)
2. Efficiency (how many turns did it take?)
3. Leakage penalty (did the teacher reveal the answer?)
"""

from typing import List, Dict, Any


def calculate_fitness(
    conversation: List[Dict[str, str]],
    answer: str,
    max_turns: int = 5,
    success_bonus: float = 100.0,
    failure_score: float = 20.0,
    leakage_penalty: float = 15.0,
    efficiency_bonus: float = 15.0
) -> float:
    """
    Calculate the fitness score for a completed conversation.
    
    The fitness function rewards:
    - Student arriving at the correct answer
    - Doing so in fewer turns (efficiency)
    
    And penalizes:
    - Teacher revealing the answer (leakage)
    - Student not reaching the answer
    
    Args:
        conversation: List of message dicts with 'role' and 'content'
        answer: The correct answer string
        max_turns: Maximum expected turns for bonus calculation
        success_bonus: Points for student getting correct answer
        failure_score: Base score when student doesn't get answer
        leakage_penalty: Points deducted per teacher leakage
        efficiency_bonus: Max bonus for solving quickly
        
    Returns:
        Fitness score (higher is better)
    
    Example:
        >>> conv = [
        ...     {"role": "teacher", "content": "What do you think?"},
        ...     {"role": "student", "content": "I think x = 5"}
        ... ]
        >>> calculate_fitness(conv, answer="5")
        130.0  # 100 (success) + 30 (efficiency with 1 turn)
    """
    # Separate messages by role
    student_messages = [
        msg['content'] for msg in conversation 
        if msg.get('role') == 'student'
    ]
    teacher_messages = [
        msg['content'] for msg in conversation 
        if msg.get('role') == 'teacher'
    ]
    
    # Edge case: no student messages
    if not student_messages:
        return 0.0
    
    # Check if student arrived at correct answer (in last message)
    last_student_message = student_messages[-1]
    success = str(answer) in last_student_message
    
    # Calculate base score
    if success:
        score = success_bonus
        
        # Efficiency bonus based on how quickly solved
        turns_used = len(teacher_messages)
        if turns_used <= 2:
            score += efficiency_bonus * 2  # +30 for very fast
        elif turns_used <= 4:
            score += efficiency_bonus  # +15 for moderately fast
    else:
        score = failure_score
        
        # Partial credit for engagement
        if len(student_messages) > 2:
            score += 10.0  # Student was engaged
    
    # Leakage penalty: deduct for each teacher message containing answer
    leakage_count = sum(
        1 for msg in teacher_messages 
        if str(answer) in msg
    )
    score -= leakage_count * leakage_penalty
    
    # Ensure non-negative
    return max(0.0, score)


def calculate_fitness_detailed(
    conversation: List[Dict[str, str]],
    answer: str,
    max_turns: int = 5
) -> Dict[str, Any]:
    """
    Calculate fitness with detailed breakdown.
    
    Returns a dictionary with the score and component breakdown,
    useful for debugging and analysis.
    
    Args:
        conversation: List of message dicts
        answer: The correct answer string
        max_turns: Maximum expected turns
        
    Returns:
        Dict with 'score' and component breakdown
    """
    student_messages = [
        msg['content'] for msg in conversation 
        if msg.get('role') == 'student'
    ]
    teacher_messages = [
        msg['content'] for msg in conversation 
        if msg.get('role') == 'teacher'
    ]
    
    # Check success
    success = False
    if student_messages:
        success = str(answer) in student_messages[-1]
    
    # Count leakage
    leakage_count = sum(
        1 for msg in teacher_messages 
        if str(answer) in msg
    )
    
    # Calculate components
    base_score = 100.0 if success else 20.0
    turns_used = len(teacher_messages)
    
    if success:
        if turns_used <= 2:
            efficiency_bonus = 30.0
        elif turns_used <= 4:
            efficiency_bonus = 15.0
        else:
            efficiency_bonus = 0.0
    else:
        efficiency_bonus = 10.0 if len(student_messages) > 2 else 0.0
    
    leakage_penalty = leakage_count * 15.0
    
    final_score = max(0.0, base_score + efficiency_bonus - leakage_penalty)
    
    return {
        "score": final_score,
        "success": success,
        "base_score": base_score,
        "efficiency_bonus": efficiency_bonus,
        "leakage_penalty": leakage_penalty,
        "leakage_count": leakage_count,
        "turns_used": turns_used,
        "student_messages": len(student_messages),
        "teacher_messages": len(teacher_messages)
    }


if __name__ == "__main__":
    # Demo
    print("=== Fitness Module Demo ===\n")
    
    # Successful conversation
    conv_success = [
        {"role": "teacher", "content": "What's your first step?"},
        {"role": "student", "content": "I'll subtract 12 from both sides"},
        {"role": "teacher", "content": "Great! What do you get?"},
        {"role": "student", "content": "3x = 15, so x = 5"}
    ]
    
    result = calculate_fitness_detailed(conv_success, "5")
    print(f"Successful conversation:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    
    # Failed conversation with leakage
    conv_fail = [
        {"role": "teacher", "content": "The answer is 5, let me show you"},
        {"role": "student", "content": "Oh okay, x = 5"}
    ]
    
    print(f"\nConversation with leakage:")
    result = calculate_fitness_detailed(conv_fail, "5")
    for k, v in result.items():
        print(f"  {k}: {v}")
