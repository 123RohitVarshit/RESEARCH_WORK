#!/usr/bin/env python3
"""
EPT Evolution Runner

Main entry point for running evolutionary topology optimization.

This script:
1. Sets up mock libraries for pedagogicalrl
2. Loads environment variables
3. Initializes the Hydra config
4. Runs the evolutionary algorithm (async for speed)

Usage:
    python run_evolution.py --config-name Qwen2.5-7B-Instruct.yaml
    
    # With custom model:
    python run_evolution.py \\
        --config-name Qwen2.5-7B-Instruct.yaml \\
        teacher_model.model_name_or_path="meta-llama/llama-3.1-70b-instruct"
"""

import os
import sys
import asyncio
import logging
import time
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
from ept.evolution import run_evolution_async, evaluate_topology_async, DEFAULT_PROBLEMS

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
# Async Main Function
# =============================================================================

async def run_experiment(cfg: DictConfig) -> None:
    """
    Run the full experiment: baselines + evolution (all async).
    
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
    
    # Use standalone EPT classroom with multi-provider fallback
    # Providers are auto-configured from .env (Groq → Cerebras → OpenRouter)
    from ept.classroom import Classroom
    
    classroom = Classroom()
    
    # =========================================================================
    # BASELINE EVALUATION: Test standard teaching methods (async)
    # =========================================================================
    from ept.topology import Topology
    
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
    logger.info("PHASE 1: BASELINE EVALUATION (ASYNC)")
    logger.info("="*60)
    
    baseline_start = time.time()
    
    # Evaluate ALL baselines concurrently
    baseline_names = list(BASELINES.keys())
    baseline_topologies = list(BASELINES.values())
    
    baseline_scores = await asyncio.gather(
        *[
            evaluate_topology_async(
                topology=topology,
                problems=PROBLEMS,
                classroom=classroom,
                max_turns=EVOLUTION_CONFIG["max_turns"],
            )
            for topology in baseline_topologies
        ]
    )
    
    baseline_results = {}
    for name, topology, score in zip(baseline_names, baseline_topologies, baseline_scores):
        topology.fitness = score
        baseline_results[name] = {"topology": topology, "score": score}
        logger.info(f"  {name}: {score:.1f}")
    
    baseline_time = time.time() - baseline_start
    logger.info(f"  Baseline evaluation: {baseline_time:.1f}s")
    
    best_baseline_name = max(baseline_results, key=lambda k: baseline_results[k]["score"])
    best_baseline_score = baseline_results[best_baseline_name]["score"]
    
    # =========================================================================
    # EVOLUTION: Optimize teaching strategy (async)
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("PHASE 2: EVOLUTIONARY OPTIMIZATION (ASYNC)")
    logger.info("="*60)
    
    generation_config = cfg.get("generation", None)
    
    results = await run_evolution_async(
        classroom=classroom,
        generation_config=generation_config,
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
    
    # Timing summary
    total_time = results.get("total_time", 0) + baseline_time
    logger.info(f"\n  Total wall-clock time:                {total_time:.1f}s")
    logger.info(f"  Baseline eval time:                   {baseline_time:.1f}s")
    if results.get("history", {}).get("gen_times"):
        avg_gen = sum(results["history"]["gen_times"]) / len(results["history"]["gen_times"])
        logger.info(f"  Avg generation time:                  {avg_gen:.1f}s")
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
        "evolution_history": results["history"],
        "timing": {
            "total_seconds": total_time,
            "baseline_seconds": baseline_time,
            "mode": "async",
        }
    }
    
    with open("evolution_results.json", "w") as f:
        json.dump(results_data, f, indent=2)
    logger.info(f"\nResults saved to: evolution_results.json")


# =============================================================================
# Hydra Entry Point
# =============================================================================

@hydra.main(config_path="config/eval", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entry point with Hydra configuration.
    
    Args:
        cfg: Hydra DictConfig object
    """
    asyncio.run(run_experiment(cfg))


if __name__ == "__main__":
    main()
