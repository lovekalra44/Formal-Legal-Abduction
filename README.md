# Formal Legal Abduction: Abductive Legal Reasoning Pipeline & NLI Evaluation

This repository contains the source code and evaluation scripts for the MSc Computing Science dissertation titled **"Formal Legal Abduction: Extracting Implicit Assumptions in Supreme Court FIRAC Appeals"**, conducted at the University of Glasgow.

## Repository Structure

* `Abductive-Pipeline-v6-final.py`: The core automated verification and abductive inference pipeline. It handles context serialization of structured FIRAC representations, high-sensitivity entailment auditing against cumulative state $\mathcal{K}_{i-1}$, deterministic JSON schema validation, and hypothesis projection into canonical First-Order Logic (FOL) predicate triples.
* `nli_eval.py`: The semantic alignment verification script utilizing the cross-encoder Natural Language Inference model (`cross-encoder/nli-deberta-v3-base`) to evaluate the qualitative legal fidelity of auto-formalized canonical triples against human expert ground-truth premises.

## System Requirements & Environment Setup

* **Language:** Python 3.9+
* **Core Dependencies:**
  * `transformers` (Hugging Face)
  * `torch`
  * `scikit-learn`
  * `sentence-transformers`
  * `pydantic` (for deterministic JSON schema enforcement)

To install the required environment, run:
```bash
pip install transformers torch scikit-learn sentence-transformers pydantic

Execution Instructions
1. Run the Abductive Verification Pipeline:
Execute the core pipeline script to process structured FIRAC benchmark inputs using the qwen-2.5-coder-7b-instruct model backend with deterministic decoding parameters ($\tau = 0.0$ = 0.0):
    python Abductive-Pipeline-v6-final.py

2. Run the Cross-Encoder NLI Evaluation:
Execute the semantic alignment script to verify gold-standard manual cohorts against pipeline-verbalized triples:
    python nli_eval.py

Author: 
Love Kalra (Student ID: 3107255K)
Institution: School of Computing Science, University of Glasgow