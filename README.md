# ML Lab

A personal learning lab for exploring **machine learning, deep learning, and AI**. This repository is organized by **concept** (regression, classification, neural networks, LLMs, etc.) to mirror how these topics interconnect.

## Quick Start

### 1. Set up Python environment

```bash
/opt/homebrew/bin/python3.12 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

### 2. Explore topics

Navigate to any topic folder and explore:

```bash
cd foundations              # Start here: math & Python basics
cd machine_learning         # Linear models, trees, sklearn
cd neural_networks          # Neural networks & deep learning
cd generative_ai            # GANs, diffusion, transformers
cd llm                      # Large language models
cd rag                      # Retrieval-augmented generation
cd agents                   # Autonomous agents
```

Each topic folder has:

- **`labs/`** — runnable scripts and examples
- **`notebooks/`** — interactive Jupyter explorations
- **`README.md`** — topic overview and prerequisites
- **`common/`** — shared utilities for that topic
- **`data/`** — example datasets

### 3. Run examples

```bash
# Run a script
python machine_learning/labs/1_gradient_descent_1d.py

# Run a notebook
jupyter notebook foundations/python/1_oop.ipynb
```

## Learning Path (Suggested)

1. **Foundations** — Math & Python fundamentals
2. **Machine Learning** — Linear models, trees, classical algorithms
3. **Neural Networks** — MLPs, backprop, first deep learning models
4. **Generative AI** — VAE, GAN, diffusion, transformers
5. **LLMs** — Large language models, attention, transformers at scale
6. **RAG** — Retrieval-augmented generation with vector databases
7. **Agents** — Autonomous agents, reasoning, tool use

## Key Conventions

- **Scripts in `labs/`** are numbered (e.g., `1_gradient_descent_1d.py`) and runnable directly
- **Notebooks in `notebooks/`** are for exploration and interactive learning
- **Shared utilities** live in each topic's `labs/common/` folder
- **Data files** in `labs/data/` use relative paths for reproducibility
- Each topic has a `README.md` explaining learning goals and prerequisites

## Resources & References

See each topic's `README.md` for curated resources. Popular ones:

- [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course)
- [deeplearning.ai Specializations](https://learn.deeplearning.ai/)
- [Hands-On Machine Learning](https://github.com/ageron/handson-ml3) (Aurélien Géron)
- [The Hundred-Page ML Book](http://themlbook.com/)
