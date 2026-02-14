# Evolutionary Pedagogical Topologies (EPT)

A research framework for evolving optimal AI teaching strategies using genetic algorithms. EPT separates the **reasoning structure** (Genotype) from the **text generation** (Phenotype) to discover effective tutoring approaches.

## 🎯 Key Innovation

Traditional RLHF fine-tuning often leads to **mode collapse** — the model converges to a single repetitive teaching script. EPT solves this by:

1. **Evolving Structure**: Optimize the sequence of pedagogical actions (diagnose → scaffold → hint → verify)
2. **Preserving Diversity**: Genetic algorithms maintain population diversity
3. **Separating Concerns**: The LLM generates text, but EPT controls the *strategy*

## 📊 Latest Results

EPT discovers hybrid teaching strategies that significantly outperform fixed baselines:

| Strategy | Genes | Fitness |
|----------|-------|---------|
| Direct Instruction | `[scaffold, scaffold, scaffold, scaffold]` | 10.0 |
| Chain-of-Thought | `[hint, hint, hint, hint]` | 56.7 |
| Verification Focus | `[verify, verify, verify, verify]` | 51.7 |
| **EPT Evolved** | **`[diagnose, verify, encourage, hint]`** | **93.3** |

**Improvement: +64.7% over best baseline (Chain-of-Thought)**

The evolved strategy follows a pedagogically sound approach: diagnose what the student knows → verify their work → encourage progress → give targeted hints.

## ⚡ Performance Features

### Async API Calls
All LLM API calls run **concurrently** via `asyncio.gather()`. Instead of evaluating topologies one by one, the entire population is evaluated in parallel:

```
BEFORE: 60 sequential API calls → ~500s per generation
AFTER:  60 concurrent API calls → ~30s per generation (~4x speedup)
```

### Multi-Provider Fallback
The Classroom automatically chains multiple free LLM providers with automatic failover:

```
Groq (fastest, 14400 req/day) → Cerebras (1M tokens/day) → OpenRouter (fallback)
```

If a provider returns a rate limit error (HTTP 429), the next one is tried automatically. All providers use the **OpenAI-compatible API format**, so no code branching is needed.

| Provider | Teacher Model | Student Model | Free Tier |
|----------|--------------|---------------|-----------|
| **Groq** | `llama-3.3-70b-versatile` | `llama-3.3-70b-versatile` | 14,400 req/day |
| **Cerebras** | `qwen-3-32b` | `llama3.1-8b` | 1M tokens/day |
| **OpenRouter** | `llama-3.3-70b-instruct:free` | `qwen3-30b-a3b:free` | 1,000 req/day |

## 🏗️ Project Structure

```
RESEARCH_WORK/
├── ept/                    # Core EPT library
│   ├── topology.py         # Genotype: Teaching strategy genes
│   ├── classroom.py        # LLM orchestrator (async + multi-provider)
│   ├── evolution.py        # Genetic algorithm (sync + async)
│   ├── fitness.py          # Scoring function
│   └── utils.py            # Selection algorithms, diversity metrics
├── mocks/                  # Lightweight mocks for heavy dependencies
│   ├── torch_mock.py       # PyTorch mock (saves 2GB)
│   ├── transformers_mock.py # HuggingFace mock (saves 500MB)
│   └── ...                 # vLLM, DeepSpeed, etc.
├── config/
│   └── eval/
│       └── Qwen2.5-7B-Instruct.yaml  # Hydra config
├── run_evolution.py        # Main entry point (async)
├── requirements.txt        # Python dependencies
└── evolution_results.json  # Latest results
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file with one or more provider keys:
```env
# Primary (fastest) — https://console.groq.com
GROQ_API_KEY=gsk_your-key-here

# Fallback — https://cloud.cerebras.ai
CEREBRAS_API_KEY=csk-your-key-here

# Last resort — https://openrouter.ai/keys
OPENROUTER_API_KEY=sk-or-your-key-here
```

At least one key is required. The more keys you add, the more resilient the system becomes.

### 3. Run Evolution

```bash
python run_evolution.py --config-name Qwen2.5-7B-Instruct.yaml
```

The console will show:
- Provider chain being used
- Per-generation timing
- Baseline vs evolved strategy comparison
- Results saved to `evolution_results.json`

## 🧬 How EPT Works

### Genotype (Topology)
A sequence of pedagogical actions — the "DNA" of a teaching strategy:
- `diagnose` — Ask what the student already knows
- `scaffold` — Break the problem into smaller steps
- `hint` — Give conceptual guidance without revealing the answer
- `verify` — Ask the student to check their work
- `encourage` — Positive reinforcement

### Phenotype (Conversation)
The LLM receives each action as a system instruction:
```
STRATEGY: Ask the student to verify their arithmetic.
PROBLEM: Solve for x: 3x + 12 = 27
```

### Fitness Function
```
+100   Student reaches correct answer
+30    Solved in ≤2 turns (efficiency bonus)
+15    Solved in ≤4 turns
 -15   Per teacher answer leak (penalty)
```

### Genetic Operators
- **Mutation**: Randomly change one action in the strategy
- **Crossover**: Combine two parent strategies at a random split
- **Selection**: Fitness-proportional (roulette wheel)
- **Elitism**: Best strategy preserved across generations

## 📁 Configuration

Evolution parameters in `run_evolution.py`:
```python
EVOLUTION_CONFIG = {
    "population_size": 4,      # Individuals per generation
    "generations": 4,          # Evolutionary generations
    "gene_length": 4,          # Strategy length (number of actions)
    "max_turns": 5,            # Max conversation turns
    "mutation_rate": 0.6,      # Mutation vs crossover probability
    "elite_count": 1,          # Elites preserved per generation
}
```

## � Lightweight Mock System (No GPU Required)

The mock system enables running on any machine without PyTorch, vLLM, or DeepSpeed:

| Mock | Replaced Library | Saves |
|------|------------------|-------|
| `torch_mock.py` | PyTorch | ~2GB |
| `transformers_mock.py` | HuggingFace Transformers | ~500MB |
| `vllm_mock.py` | vLLM inference engine | GPU requirement |
| `deepspeed_mock.py` | DeepSpeed | GPU requirement |

## 🔬 Research Roadmap

- [ ] Integrate GSM8K benchmark (50+ problems)
- [ ] Multi-run evaluation with statistical significance
- [ ] Ablation studies (mutation, crossover, population size)
- [ ] Robust evaluation (multi-sample averaging for noise reduction)
- [ ] Cross-model transfer analysis
- [ ] Human evaluation study

## 📄 License

MIT License — See LICENSE file for details.

## 🙏 Acknowledgments

- Based on the [pedagogicalrl](https://github.com/eth-lre/pedagogicalrl) framework
- LLM inference via [Groq](https://groq.com), [Cerebras](https://cerebras.ai), and [OpenRouter](https://openrouter.ai)
- Configuration management with [Hydra](https://hydra.cc/)
