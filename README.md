# Fine-Tuning LLMs for Text-to-SQL Generation
### A Fraud Analytics Perspective — IIT Delhi GenAI Capstone (Distinction, 92/100)

This project fine-tunes **Qwen2-0.5B-Instruct** to convert natural language fraud investigation questions into executable SQL queries. Four parameter-efficient fine-tuning (PEFT) methods are compared across 13 evaluation metrics on a domain-filtered Text-to-SQL dataset.

---

## Problem Statement

Fraud analysts at fintech organisations must translate investigative hypotheses — velocity anomalies, geographic impossibilities, behavioural deviations — into precise SQL queries. This process is time-consuming and requires significant technical expertise. This project explores whether a small, fine-tuned LLM can bridge that gap.

---

## Approach

| Method | Trainable Parameters | Description |
|---|---|---|
| Full Fine-Tuning | 494M (100%) | Performance ceiling; all weights updated |
| LoRA | 8.8M (1.75%) | Low-rank adapter matrices injected into attention layers |
| QLoRA | 8.8M (1.75%) | 4-bit quantized base model + LoRA adapters |
| Prefix Tuning | 61K (0.012%) | Learnable virtual tokens prepended to input |

**Model:** Qwen2-0.5B-Instruct  
**Dataset:** [gretelai/synthetic_text_to_sql](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql) — filtered for finance, insurance, cybersecurity, legal, and blockchain domains  
**Training:** NVIDIA A100 (40GB VRAM) via Google Colab Pro  

---

## Key Results

| Method | BLEU | ROUGE-1 | BERT Score | Runtime |
|---|---|---|---|---|
| Full Fine-Tuning | 0.4759 | 0.8121 | 0.9565 | ~35 min |
| **LoRA** | **0.4533** | **0.8129** | **0.9555** | **~10 min** |
| QLoRA | 0.4501 | 0.8032 | 0.9538 | ~10 min |
| Prefix Tuning | 0.0005 | 0.0883 | 0.7623 | ~10 min |

**Key finding:** LoRA achieves within 3% of Full Fine-Tuning performance while training only 1.75% of parameters. QLoRA offers a further memory reduction with a small quality trade-off. Prefix Tuning fails for structured SQL generation — 61K parameters are insufficient to override the frozen model's natural language priors in a single epoch.

---

## Evaluation Framework

Evaluation spans 13 metrics across 4 dimensions on 500 held-out test examples:

- **Similarity metrics:** BLEU, ROUGE-1/2/L, METEOR, GLEU, CoSIM, BERT Score
- **Quality metrics:** Repetition Rate, Flesch Reading Ease, Toxicity, Novelty, Diversity
- **Training dynamics:** Loss curves across all four methods
- **Qualitative:** Side-by-side SQL output comparison across domains (financial services, justice, insurance)

---

## Repo Structure

```
├── Project_Fine_tuning_Qwen2_text_2_SQL.ipynb   # Full training + evaluation notebook
├── Project_Fine_Tuning_Presentation.pptx         # IIT Delhi capstone presentation
└── README.md
```

**Note on outputs:** Model checkpoints and generated CSVs are not included due to file size. The notebook is fully reproducible on Google Colab Pro (A100 runtime) — both the dataset and base model are publicly available with no authentication required.

---

## How to Run

1. Open the notebook in Google Colab
2. Set runtime to **A100 GPU** (Runtime → Change runtime type)
3. Run all cells sequentially — installs, dataset loading, and model downloads are handled in Section 0
4. Outputs (evaluation CSVs, plots) are saved to `/content/SQL_FT_checkpoints/` and can be downloaded at the end

---

## Fraud Domain Applicability

From a production fraud analytics perspective, **LoRA is the recommended deployment choice**:

- Performance within 3% of Full Fine-Tuning across all metrics
- 35MB adapter file vs 1.97GB full model — trivial to version, deploy, and update
- Retrainable on new fraud typologies (card fraud, AML, identity fraud) in ~10 minutes
- Multiple domain-specific adapters can be hot-swapped onto the same frozen base model

---

## Technical Stack

`Python` `PyTorch` `Transformers (HuggingFace)` `PEFT` `TRL` `BitsAndBytes` `Qwen2` `Google Colab`

---

## About

Built as part of the **IIT Delhi Certificate Programme in Generative AI** (Distinction, 92/100).  
Author: Yoshita Gureja | [LinkedIn](https://www.linkedin.com/in/YOUR-LINKEDIN-HANDLE)  
Domain: Fraud Risk Strategy × Generative AI
