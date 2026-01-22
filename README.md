# Evolutionary Pedagogical Topologies (EPT)

A research framework for evolving optimal teaching strategies using genetic algorithms. EPT separates the **reasoning structure** (Genotype) from the **text generation** (Phenotype) to discover effective tutoring approaches.

## 🎯 Key Innovation

Traditional RLHF fine-tuning often leads to **mode collapse** - the model converges to a single repetitive teaching script. EPT solves this by:

1. **Evolving Structure**: Optimize the sequence of pedagogical actions (diagnose → scaffold → hint → verify)
2. **Preserving Diversity**: Genetic algorithms maintain population diversity
3. **Separating Concerns**: The LLM generates text, but EPT controls the *strategy*

## 📊 Results

EPT demonstrates significant improvement over standard teaching baselines:

| Strategy | Genes | Fitness |
|----------|-------|---------|
| Direct Instruction | `[scaffold, scaffold, scaffold, scaffold]` | 56.7 |
| Chain-of-Thought | `[hint, hint, hint, hint]` | 51.7 |
| Verification Focus | `[verify, verify, verify, verify]` | 93.3 |
| **EPT Evolved** | `[verify, scaffold, verify, verify]` | **125.0** |

**Improvement: +34% over best baseline**

The evolved strategy combines multiple approaches rather than repeating a single action, discovering a hybrid teaching method.

## 🏗️ Project Structure

```
RESEARCH_WORK/
├── ept/                    # Core EPT library
│   ├── topology.py         # Genotype: Teaching strategy genes
│   ├── classroom.py        # Phenotype: LLM conversation wrapper
│   ├── evolution.py        # Genetic algorithm loop
│   ├── fitness.py          # Scoring function
│   └── utils.py            # Selection algorithms, diversity metrics
├── mocks/                  # Lightweight mocks for heavy dependencies
│   ├── torch_mock.py       # PyTorch mock (saves 2GB)
│   ├── transformers_mock.py # HuggingFace mock (saves 500MB)
│   └── ...                 # vLLM, DeepSpeed, etc.
├── config/
│   └── eval/
│       └── Qwen2.5-7B-Instruct.yaml  # Hydra config
├── run_evolution.py        # Main entry point
├── requirements.txt        # Python dependencies
└── evolution_results.json  # Latest results
```

## 💡 Lightweight Mock System (No GPU Required)

A key innovation of this project is the **mock system** that enables running on any machine without heavy dependencies:

### The Problem
The original pedagogicalrl framework requires:
- **PyTorch**: ~2GB installation
- **Transformers**: ~500MB installation  
- **vLLM**: Requires CUDA GPU
- **DeepSpeed**: Requires CUDA GPU

This makes local development and testing nearly impossible without expensive GPU hardware.

### The Solution
We created lightweight mocks that provide the same interfaces without the heavy dependencies:

| Mock | Replaced Library | Saves |
|------|------------------|-------|
| `torch_mock.py` | PyTorch | ~2GB |
| `transformers_mock.py` | HuggingFace Transformers | ~500MB |
| `vllm_mock.py` | vLLM inference engine | GPU requirement |
| `deepspeed_mock.py` | DeepSpeed | GPU requirement |
| `pynvml_mock.py` | NVIDIA GPU monitoring | GPU requirement |

### How It Works
```python
# Before any imports, set up mocks
from mocks import setup_all_mocks
setup_all_mocks()

# Now safe to import - uses mocks instead of real libraries
from ept.classroom import Classroom
```

### API-Based Inference
Instead of local GPU inference, we use **OpenRouter API** for LLM calls:
- Works on any machine (laptop, cloud, etc.)
- No GPU required
- Pay-per-use pricing (~$0.001 per 1K tokens)

This approach makes the research accessible to anyone, regardless of hardware.

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

Create a `.env` file:
```
OPENROUTER_API_KEY=your-key-here
```

Get your API key from [OpenRouter](https://openrouter.ai/).

### 3. Run Evolution

```bash
python run_evolution.py --config-name Qwen2.5-7B-Instruct.yaml
```

## 📈 Understanding the Output

The evolution runs in two phases:

### Phase 1: Baseline Evaluation
Tests standard teaching approaches:
- **Direct Instruction**: Repeatedly break down problems
- **Chain-of-Thought**: Repeatedly give hints
- **Verification Focus**: Repeatedly ask student to verify

### Phase 2: Evolutionary Optimization
- Population of 4 teaching strategies
- 4 generations of evolution
- Mutation rate: 60%
- Elitism: Best strategy preserved

### Results Comparison
```
============================================================
IMPROVEMENT SUMMARY
------------------------------------------------------------
  Best Baseline (Verification Focus): 93.3
  Evolved Strategy:                   125.0
  Absolute Improvement:               +31.7
  Relative Improvement:               +34%
============================================================
```

## 🧬 How EPT Works

### Genotype (Topology)
A sequence of pedagogical actions:
- `diagnose` - Ask what the student thinks
- `scaffold` - Break problem into steps
- `hint` - Give conceptual guidance
- `verify` - Ask student to check work
- `encourage` - Positive reinforcement

### Phenotype (Conversation)
The LLM receives the action as an instruction:
```
STRATEGY: Ask the student to verify their arithmetic.
PROBLEM: Solve for x: 3x + 12 = 27
```

### Fitness Function
```
+100  Student gets correct answer
+30   Solved in ≤2 turns (efficiency bonus)
+15   Solved in ≤4 turns
-15   Per teacher answer leak (penalty)
```

### Genetic Operators
- **Mutation**: Randomly change one gene
- **Crossover**: Combine two parent strategies
- **Selection**: Fitness-proportional (roulette wheel)

## 📁 Configuration

Edit `config/eval/Qwen2.5-7B-Instruct.yaml`:

```yaml
teacher_model:
  model_name_or_path: "meta-llama/llama-3.1-8b-instruct"
  use_openrouter: true

student_model:
  model_name_or_path: "meta-llama/llama-3.1-8b-instruct"
  use_openrouter: true
```

Edit evolution parameters in `run_evolution.py`:
```python
EVOLUTION_CONFIG = {
    "population_size": 4,
    "generations": 4,
    "gene_length": 4,
    "max_turns": 5,
    "mutation_rate": 0.6,
}
```

## 📝 Customizing Problems

Edit the problems list in `run_evolution.py`:
```python
PROBLEMS = [
    {"problem": "Solve for x: 3x + 12 = 27", "answer": "5"},
    {"problem": "Solve for y: 2y - 8 = 10", "answer": "9"},
    # Add more problems...
]
```

## 🔬 Research Applications

- Compare teaching strategies across different student models
- Evolve domain-specific tutoring approaches
- Study diversity maintenance in pedagogical evolution
- Analyze optimal action sequences for different problem types

## 📄 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- Based on the [pedagogicalrl](https://github.com/eth-lre/pedagogicalrl) framework
- Uses [OpenRouter](https://openrouter.ai/) for LLM inference
- Built with [Hydra](https://hydra.cc/) for configuration management
