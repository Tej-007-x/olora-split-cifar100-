# O-LoRA for Continual Learning: A CNN Adaptation on Split-CIFAR100

**Course Project** — Deep Learning / Continual Learning
**Supervisor:** Prof. Antonio Carta, University of Pisa
**Student:** Teja Karuku
**Matricola:** 724955

---

## 1. Objective

Neural networks suffer from **catastrophic forgetting**: when trained sequentially on a series of tasks, they tend to overwrite the Iights that encoded earlier tasks while learning new ones.

This project implements and evaluates **O-LoRA (Orthogonal Low-Rank Adaptation)**, proposed in:

> Wang, X., Chen, T., Ge, Q., Xia, H., Bao, R., Zheng, R., Zhang, Q., Gui, T., & Huang, X. (2023). *Orthogonal Subspace Learning for Language Model Continual Learning.* arXiv:2310.14152.

The original paper targets **Transformer-based language models** (T5, LLaMA) on **text classification** benchmarks, applying LoRA adapters to attention Q/V projection matrices and enforcing orthogonality between each new task's adapter and all previous tasks' adapters.

**My contribution:** I adapt the same core idea — task-specific low-rank adapters kept orthogonal to one another — to a **CNN (ResNet-18)** operating on **images (Split-CIFAR100)**, and empirically test whether the paper's central hypothesis holds in this different domain.

---

## 2. Core Hypothesis Being Tested

> Does constraining each new task's low-rank adapter to be orthogonal to previous tasks' adapters reduce catastrophic forgetting compared to unconstrained sequential adaptation?

I test this by training two versions of the same pipeline:
- **No-Orthogonality baseline** (λ₁ = 0): sequential LoRA adapters, no constraint.
- **O-LoRA** (λ₁ = 0.1): same setup, with the orthogonality regularization term active.

---

## 3. Method Summary

| Component | Paper (NLP) | My CNN Adaptation |
|---|---|---|
| Backbone | T5 / LLaMA (frozen) | ResNet-18, ImageNet-pretrained (frozen) |
| Adapted layers | Attention Q/V projections | Every 3×3 Conv2d inside ResNet BasicBlocks |
| Task structure | 5–15 NLP classification tasks | 10 sequential tasks, 10 CIFAR-100 classes each |
| Adapter | `ΔW = A·B`, A ∈ R^(d×r), B ∈ R^(r×k) | Same decomposition, applied to flattened conv kernels |
| Orthogonality | `O_{i,t} = Aᵢᵀ·Aₜ`, penalize Frobenius norm | Identical formula, per conv layer, averaged across layers |
| Task-ID at inference | Not required (paper's stated advantage) | Not required — each finished task's adapter is **merged** into the frozen backbone weights permanently after training |
| Classifier head | N/A (generative LM) | Single shared `Linear` layer over all 100 classes; loss/eval restricted to the current task's class-column slice (task-incremental protocol) |

### Key design decisions worth noting for review

1. **Weight merging (paper Eq. 9):** after each task finishes training, its adapter's delta is folded directly into the frozen conv Iights. This avoids growing adapter storage indefinitely and — more importantly for evaluation — means no task-ID routing is needed at inference time, matching the paper's generalization-friendliness claim.
2. **Task-incremental class-slicing:** the classifier is a single shared layer over all 100 classes, but both training loss and evaluation accuracy are computed only on the 10 output columns belonging to the task in question. This isolates *backbone-level* forgetting (what O-LoRA targets) from an unrelated *classifier recency-bias* artifact that would otherwise dominate and mask the effect being studied.
3. **Orthogonality loss shape:** implemented as `Aᵢᵀ @ Aₜ` → an `(r × r)` matrix (r = LoRA rank), matching the paper's Eq. 6 exactly — not a full feature-dimension matrix, which would be both mathematically incorrect and computationally wasteful.

---

## 4. Repository / Notebook Structure

| Cell | Contents |
|---|---|
| 1 | Install dependencies |
| 2 | Imports |
| 3 | Global `Config` (hyperparameters, paths) |
| 4 | Seeding & device setup |
| 5 | Google Drive mount + CIFAR-100 archive extraction |
| 6 | Data transforms + dataset loading |
| 7 | `SplitCIFAR100Benchmark` — builds the 10-task split |
| 8 | `OLoRASubspace` / `OLoRAConv2d` — the adapter + merge logic |
| 9 | `OLoRAResNet18` — full model with adapters injected |
| 10 | `OrthogonalityLoss` — Eq. 6–8 implementation |
| 11 | Optimizer / scheduler builders |
| 12 | Training loop (`train_one_task`) and evaluation (`evaluate_task`) |
| 13 | `run_olora_pipeline` — reusable full-sequence runner |
| 14 | Main experiment: runs both λ=0 and λ=0.1 configurations |
| 15 | Saves accuracy matrices to Drive |
| 16 | Generates comparison plots (heatmaps + forgetting curve) |

---

## 5. How to Run

1. Upload `cifar-100-python.tar.gz` (official CIFAR-100 archive) to your Google Drive.
2. Open the notebook in Google Colab, set runtime to **GPU (T4)**.
3. Run all cells top to bottom. Cell 5 will locate the archive automatically anywhere in your Drive.
4. Cell 14 runs the full experiment (~2× a single 10-task pass, since it runs both configurations).
5. Results (accuracy matrices, comparison plot) are saved to `results_dir` in Drive.

---

## 6. Results

| Method | Average Accuracy (AA) | Average Forgetting (AF) |
|---|---|---|
| No Orthogonality (λ=0) | 40.72% | 26.51% |
| **O-LoRA (λ=0.1)** | **44.08%** | **22.27%** |

- **Forgetting reduced by 4.24 percentage points**
- **Accuracy improved by 3.36 percentage points**

Both runs show the expected monotonic decline in older tasks' accuracy as more tasks are learned, and O-LoRA consistently forgets less than the unconstrained baseline across the sequence — supporting the paper's core hypothesis in this CNN/image setting.

---

## 7. Metrics Used

Following the paper's Average Accuracy definition (Section 4.1.2):

$$AA = \frac{1}{T}\sum_{i=1}^{T} a_{i,T}$$

where `a_{i,j}` is the test accuracy on task *i* after training on task *j*.

**Average Forgetting** (not explicitly reported as a scalar in the original paper, but standard in the continual learning literature) is computed as:

$$AF = \frac{1}{T-1}\sum_{i=1}^{T-1} \left(\max_j a_{i,j} - a_{i,T}\right)$$

---

## 8. Limitations & Honest Caveats

- **Not directly comparable to the paper's reported numbers.** The paper reports ~75.8% AA on T5-large across 5 NLP tasks — a different model, domain, and task count. My result should be read as a **relative** comparison (with vs. without orthogonality) within My own CNN setup, not an absolute benchmark against the paper.
- **Single seed, single run.** Results here come from one training run per configuration. For a more rigorous claim, the same experiment should be repeated across 2–3 seeds and averaged (as the paper does).
- **Limited training budget.** Due to free-tier Colab GPU constraints, each task is trained for only 1–2 epochs with a modest LoRA rank (8–16). A longer training budget would likely improve both methods' absolute accuracy.
- **Effect size is modest but consistent.** The ~4-point forgetting reduction is smaller than the paper's NLP results, which is expected given the much smaller adapted model (ResNet-18, 11M params) and image domain versus the paper's large pretrained language models.

---

## 9. Conclusion

The orthogonality constraint from O-LoRA transfers, at least directionally, from Transformer/NLP settings to a CNN/image setting: constraining new task adapters to be orthogonal to prior ones reduces catastrophic forgetting and improves overall accuracy on Split-CIFAR100, consistent with the paper's central claim.
