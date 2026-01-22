#!/usr/bin/env python3
"""
EPT Evolution Runner

Main entry point for running evolutionary topology optimization.

This script:
1. Sets up mock libraries for pedagogicalrl
2. Loads environment variables
3. Initializes the Hydra config
4. Runs the evolutionary algorithm

Usage:
    python run_evolution.py --config-name Qwen2.5-7B-Instruct.yaml
    
    # With custom model:
    python run_evolution.py \\
        --config-name Qwen2.5-7B-Instruct.yaml \\
        teacher_model.model_name_or_path="meta-llama/llama-3.1-70b-instruct"
"""

import os
import sys
import logging
from pathlib import Path

# =============================================================================
# Step 0: Fix Windows console encoding for Unicode
# =============================================================================
if sys.platform == "win32":
    # Enable UTF-8 mode for Windows console
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# =============================================================================
# Step 1: Setup mocks BEFORE any pedagogicalrl imports
# =============================================================================

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup mock libraries
from mocks import setup_all_mocks
setup_all_mocks()

# Add pedagogicalrl to path
pedagogicalrl_path = project_root / "pedagogicalrl"
if pedagogicalrl_path.exists():
    sys.path.insert(0, str(pedagogicalrl_path))

# =============================================================================
# Step 2: Now safe to import everything else
# =============================================================================

import hydra
from hydra.core.config_store import ConfigStore
from dotenv import load_dotenv

from omegaconf import DictConfig
from ept.evolution import run_evolution, DEFAULT_PROBLEMS

# Configure logging with UTF-8 handler
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(stream=sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Note: We do NOT register a ConfigStore schema to allow flexible YAML loading


# =============================================================================
# Evolution Parameters
# =============================================================================

# These can be modified for different experiments
EVOLUTION_CONFIG = {
    "population_size": 4,      # Number of individuals per generation
    "generations": 4,          # Number of evolutionary generations
    "gene_length": 4,          # Length of topology genes
    "max_turns": 5,            # Max conversation turns per evaluation
    "mutation_rate": 0.6,      # Probability of mutation vs crossover
    "elite_count": 1,          # Number of elites to preserve
}

# Math problems for evaluation
PROBLEMS = [
    {"problem": "Solve for x: 3x + 12 = 27", "answer": "5"},
    {"problem": "Solve for y: 2y - 8 = 10", "answer": "9"},
    {"problem": "Solve for z: 5z + 3 = 18", "answer": "3"},
]


# =============================================================================
# Main Function
# =============================================================================

@hydra.main(config_path="config/eval", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entry point with Hydra configuration.
    
    Args:
        cfg: Hydra DictConfig object
    """
    # Load environment variables
    load_dotenv()
    
    # Validate API key
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("[ERROR] No API key found!")
        logger.error("   Please set OPENROUTER_API_KEY or GEMINI_API_KEY in .env")
        sys.exit(1)
    
    logger.info("[OK] API key loaded successfully")
    
    # Use standalone EPT classroom (no pedagogicalrl dependency)
    from ept.classroom import Classroom
    
    # Get model names from config
    teacher_model = cfg.get("teacher_model", {}).get("model_name_or_path", "meta-llama/llama-3.1-8b-instruct")
    student_model = cfg.get("student_model", {}).get("model_name_or_path", "meta-llama/llama-3.1-8b-instruct")
    
    # Create classroom instance
    classroom = Classroom(
        teacher_model=teacher_model,
        student_model=student_model
    )
    
    # =========================================================================
    # BASELINE EVALUATION: Test standard teaching methods first
    # =========================================================================
    from ept.topology import Topology
    from ept.evolution import evaluate_topology
    
    # Define proper baselines representing real teaching approaches
    BASELINES = {
        "Direct Instruction": Topology(
            genes=['scaffold', 'scaffold', 'scaffold', 'scaffold'],
            fitness=-999
        ),
        "Chain-of-Thought": Topology(
            genes=['hint', 'hint', 'hint', 'hint'],
            fitness=-999
        ),
        "Verification Focus": Topology(
            genes=['verify', 'verify', 'verify', 'verify'],
            fitness=-999
        ),
    }
    
    logger.info("\n" + "="*60)
    logger.info("PHASE 1: BASELINE EVALUATION")
    logger.info("="*60)
    
    baseline_results = {}
    for name, topology in BASELINES.items():
        score = evaluate_topology(
            topology=topology,
            problems=PROBLEMS,
            classroom=classroom,
            generation_config=cfg.generation,
            max_turns=EVOLUTION_CONFIG["max_turns"]
        )
        topology.fitness = score
        baseline_results[name] = {"topology": topology, "score": score}
        logger.info(f"  {name}: {score:.1f}")
    
    best_baseline_name = max(baseline_results, key=lambda k: baseline_results[k]["score"])
    best_baseline_score = baseline_results[best_baseline_name]["score"]
    
    # =========================================================================
    # EVOLUTION: Optimize teaching strategy
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("PHASE 2: EVOLUTIONARY OPTIMIZATION")
    logger.info("="*60)
    
    results = run_evolution(
        classroom=classroom,
        generation_config=cfg.generation,
        problems=PROBLEMS,
        **EVOLUTION_CONFIG
    )
    
    evolved = results["best_topology"]
    
    # =========================================================================
    # COMPARISON: Show improvement
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("RESULTS COMPARISON")
    logger.info("="*60)
    
    logger.info("\nBASELINE STRATEGIES:")
    for name, data in baseline_results.items():
        marker = " <-- best baseline" if name == best_baseline_name else ""
        logger.info(f"  {name}: {data['score']:.1f}{marker}")
        logger.info(f"    Genes: {data['topology'].genes}")
    
    logger.info(f"\nEVOLVED STRATEGY:")
    logger.info(f"  EPT Optimized: {evolved.fitness:.1f}")
    logger.info(f"    Genes: {evolved.genes}")
    
    # Calculate improvement
    improvement = evolved.fitness - best_baseline_score
    improvement_pct = (improvement / max(best_baseline_score, 1)) * 100
    
    logger.info("\n" + "-"*60)
    logger.info("IMPROVEMENT SUMMARY")
    logger.info("-"*60)
    logger.info(f"  Best Baseline ({best_baseline_name}): {best_baseline_score:.1f}")
    logger.info(f"  Evolved Strategy:                     {evolved.fitness:.1f}")
    logger.info(f"  Absolute Improvement:                 +{improvement:.1f}")
    logger.info(f"  Relative Improvement:                 +{improvement_pct:.1f}%")
    logger.info("="*60)
    
    # Save results to JSON
    import json
    results_data = {
        "baselines": {
            name: {"genes": data["topology"].genes, "fitness": data["score"]}
            for name, data in baseline_results.items()
        },
        "evolved": {
            "genes": evolved.genes,
            "fitness": evolved.fitness
        },
        "improvement": {
            "absolute": improvement,
            "percentage": improvement_pct,
            "vs_baseline": best_baseline_name
        },
        "evolution_history": results["history"]
    }
    
    with open("evolution_results.json", "w") as f:
        json.dump(results_data, f, indent=2)
    logger.info(f"\nResults saved to: evolution_results.json")


if __name__ == "__main__":
    main()
