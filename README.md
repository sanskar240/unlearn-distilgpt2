# Surgical Isolation of a Factual Subspace in DistilGPT-2


<img width="801" height="470" alt="diff_norm" src="https://github.com/user-attachments/assets/82e65930-17ac-460d-ac11-ea683541627b" />



**Mechanistic Unlearning | Rank-1 Subspace | No SAE**

---

## Discovery
- **Bimodal factual encoding**: Facts **enter** at **Layer 1**, **compress** at **Layer 3**, **reconstruct** at **Layer 5**
- **Mean-diff subspace** isolates facts in **one vector per layer**
- **Causal pruning** kills fact recall while preserving reasoning

---

## Results
| Layer | Signal | Causal Drop |
|-------|--------|-------------|
| 1     | STRONG | −80%        |
| 3     | SILENT | 0%          |
| 5     | STRONG | −60%        |

---

## Quick Start

```bash
git clone https://github.com/yourname/factual-subspace-distilgpt2
cd factual-subspace-distilgpt2
pip install -r requirements.txt
python run.py
