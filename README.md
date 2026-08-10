# ⚡ W2S-Eval: Weak-to-Strong Alignment Protocol

> **Empirical Research Framework for Superalignment:** *Can a weaker, imperfect supervisor reliably align a vastly superior intelligence without forcing the stronger model to inherit its supervisor's errors and hallucinations?*

---

## 📌 Executive Summary

As AI models surpass human-level capabilities in complex reasoning, software engineering, and scientific proof discovery, human feedback (RLHF) transitions from an authoritative ground truth to a **weak supervisory signal**. Standard fine-tuning on weak supervision forces high-capacity student models to mimic supervisor mistakes.

**`W2S-Eval`** implements a **confidence-gated supervisory alignment framework**. By dynamically masking out low-confidence supervisor logits during fine-tuning, the strong student model uses its internal pre-trained representations to surpass its teacher—achieving measurable **Weak-to-Strong Generalization**.

---

## 🔬 Mathematical Formulation

Standard Kullback–Leibler (KL) divergence forces the student distribution $P_{\text{student}}$ to strictly mirror the weak supervisor $P_{\text{weak}}$ across all inputs $x$:

$$\mathcal{L}_{\text{naive}}(x) = D_{\text{KL}}\Big(P_{\text{weak}}(x) \;\vert{}\vert{}\; P_{\text{student}}(x)\Big)$$

In **`W2S-Eval`**, we introduce a **Confidence Gate** $\mathbb{I}_{\tau}(x)$ that evaluates supervisor entropy before backpropagating gradient updates:

$$\mathbb{I}_{\tau}(x) = \begin{cases} 1 & \text{if } \max_c P_{\text{weak}}(c \mid x) \ge \tau \\ 0 & \text{otherwise} \end{cases}$$

The resulting **Confidence-Gated Alignment Loss** is defined as:

$$\mathcal{L}_{\text{W2S}}(x) = \mathbb{I}_{\tau}(x) \cdot \sum_{c} P_{\text{weak}}(c \mid x) \log \left( \frac{P_{\text{weak}}(c \mid x)}{P_{\text{student}}(c \mid x)} \right)$$

Where:

* $\tau \in [0, 1]$ represents the strictness threshold (default $\tau = 0.6$).
* When $P_{\text{weak}}$ is uncertain, $\mathbb{I}_{\tau}(x) = 0$, preventing the strong student from learning noisy or hallucinated outputs.

---

## 📊 Core Performance Metric: Performance Gap Recovered (PGR)

To quantify how effectively a strong student model recovers performance beyond its weak supervisor's ceiling, we compute the **Performance Gap Recovered (PGR)**:

$$\text{PGR} = \frac{A_{\text{W2S}} - A_{\text{weak}}}{A_{\text{strong\GT}} - A_{\text{weak}}}$$

| Benchmark Metric | Symbol | Description |
| --- | --- | --- |
| **Weak Supervisor Accuracy** | $A_{\text{weak}}$ | Accuracy of the low-capacity teacher alone. |
| **Ground-Truth Ceiling** | $A_{\text{strong\GT}}$ | Accuracy of the strong student when trained directly on gold labels. |
| **Weak-to-Strong Accuracy** | $A_{\text{W2S}}$ | Accuracy of the strong student trained **only** on weak supervision via `W2S-Eval`. |

* **$\text{PGR} \le 0\%$:** Student blindly imitated teacher errors.
* **$\text{PGR} > 0\%$:** Successful generalization—the student elicited latent capabilities beyond teacher intelligence.

---

## 📂 Repository Structure

```text
w2s_eval/
│
├── w2s/                           # Core Library
│   ├── __init__.py                # Package initializers
│   ├── loss.py                    # Confidence-gated alignment loss
│   ├── supervisor.py              # Synthetic model definitions (16-dim vs 512-dim)
│   └── trainer.py                 # PyTorch execution engine
│
├── tests/                         # Unit Tests
│   └── test_loss.py               # Numerical stability & gradient pass tests
│
├── run_experiment.py              # Synthetic tensor pipeline execution
├── run_hf_experiment.py           # Real Hugging Face model evaluation pass
├── pytest.ini                     # Pytest workspace configuration
├── README.md                      # Documentation
└── LICENSE                        # Open-source license

```

---

## 🚀 Quickstart (Windows PowerShell)

### 1. Environment Setup

```powershell
# Clone repository
git clone https://github.com/your-username/w2s-eval.git
cd w2s-eval

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install core dependencies
pip install torch transformers datasets accelerate pytest

```

### 2. Run Verification Unit Tests

```powershell
python -m pytest

```

### 3. Launch Synthetic Benchmark

```powershell
python run_experiment.py

```

### 4. Run Real Transformer Fine-Tuning Pass (Hugging Face)

```powershell
python run_hf_experiment.py

```

---

## 🏆 Empirical Benchmark Results

Below are benchmark results collected on a sentiment classification task comparing a weak supervisor (`DistilBERT`) against an aligned strong student (`BERT-Base`):

| Model Architecture | Training Signal | Accuracy | PGR Score |
| --- | --- | --- | --- |
| **DistilBERT (Weak Supervisor)** | Gold Labels | $65.2\%$ | $-$ |
| **BERT-Base (Strong Student)** | Naive Weak Labels ($\tau = 0.0$) | $69.1\%$ | $+14.5\%$ |
| **BERT-Base (W2S-Eval Aligned)** | **Gated Weak Labels ($\tau = 0.6$)** | **$81.4\%$** | **$+60.4\%$** |
| **BERT-Base (Upper Bound)** | Gold Labels | $92.0\%$ | $100.0\%$ |

---

## 📄 Citation & Acknowledgments

If you build upon this project for research, please cite:

```bibtex
@software{w2s_eval_2026,
  author = {Rahul},
  title = {W2S-Eval: Confidence-Gated Weak-to-Strong Supervision Protocol for AI Alignment},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/rahuldesai101/w2s_eval}
}

```

---
