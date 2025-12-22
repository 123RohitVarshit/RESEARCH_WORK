***
# Evolutionary Pedagogical Topologies (EPT)

**Research Prototype | ETH Zurich Summer Fellowship Application**  
*An extension of the `eth-lre/pedagogicalrl` framework.*

## Abstract
Current Reinforcement Learning (RL) approaches for pedagogical alignment often suffer from **mode collapse**, converging to repetitive teaching scripts. This project implements **Evolutionary Pedagogical Topologies (EPT)**, a framework that separates the **Genotype** (reasoning structure) from the **Phenotype** (text execution). By evolving pedagogical strategies offline, we generate diverse, high-quality tutoring logic with significantly higher token efficiency than unstructured Chain-of-Thought.

## Technical Architecture

This prototype extends the original `pedagogicalrl` codebase with three modular components:

1.  **The Genotype (`src/topology.py`)**: Defines the DNA of a teaching strategy. Genes consist of pedagogical primitives: `DIAGNOSE`, `SCAFFOLD`, `HINT`, `VERIFY`, `ENCOURAGE`.
2.  **The Phenotype Wrapper (`src/topology_classroom.py`)**: Inherits from the base `Conversation` class. It injects dynamic instructions into the system prompt based on the current gene, enforcing structure on the LLM's generation.
3.  **The Evolutionary Engine (`run_evolution.py`)**: A Genetic Algorithm loop replacing the standard RL trainer. It uses **Fitness Proportional Selection** and **Elitism** to optimize strategies across multiple algebraic problems.

### Engineering & Optimization
To run this HPC-grade research code on standard environments (Colab/CPU), I implemented a **Runtime Mocking System**:
*   **Challenge:** The base repo requires `vllm` and `deepspeed` (heavy GPU libraries).
*   **Solution:** I implemented a `sys.modules` interception layer that mocks these libraries in memory.
*   **Inference:** Inference is routed via adapters to **OpenRouter/Gemini APIs** (using Llama-3-8B), allowing rapid iteration without local A100 GPUs.

## How to Run

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/evolutionary-pedagogy-prototype.git
    cd evolutionary-pedagogy-prototype
    ```

2.  **Install Dependencies**
    *(Note: We use lightweight dependencies only. Heavy GPU libs are mocked at runtime.)*
    ```bash
    pip install hydra-core omegaconf python-dotenv openai google-generativeai colorama
    ```

3.  **Set API Keys**
    Create a `.env` file or export variables:
    ```bash
    export OPENROUTER_API_KEY="sk-..."
    # OR
    export GEMINI_API_KEY="AIza..."
    ```

4.  **Execute the Evolution**
    ```bash
    python run_evolution.py
    ```

## Results Summary

Preliminary runs (N=4, G=4) using Llama-3-8B as the backend demonstrate strong convergence and diversity preservation:

| Metric | Result | Analysis |
| :--- | :--- | :--- |
| **Optimization** | **81.7 $\to$ 120.0** | Fitness score improved by **~47%** over 4 generations. |
| **Diversity** | **0.75** | High structural diversity indicates successful avoidance of mode collapse. |
| **Strategy** | `[Encourage, Verify...]` | The system evolved a "Socratic" verification loop rather than a lecturing style. |

## Project Structure

```text
├── run_evolution.py         # Main Genetic Algorithm loop & Mocking setup
├── src/
│   ├── topology.py          # Genotype logic & Mutation operators
│   ├── topology_classroom.py# Prompt injection & Conversation wrapper
│   └── classroom.py         # Base environment logic (Patched for API usage)
├── config/                  # Hydra configuration files
└── prompt_templates/        # Jinja2 templates for Socratic tutoring
```

## Ongoing improvements
1. Trying to structure code into proper python files.
2. Working on using SLM as evaluator and try to decrease the compute.  
