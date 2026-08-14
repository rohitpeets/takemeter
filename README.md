# TakeMeter

A fine-tuned text classifier that evaluates discourse quality in r/soccer, Reddit's largest football community. TakeMeter classifies comments into four categories — Analysis, Prediction, Reaction, and Humor — and compares a fine-tuned DistilBERT model against a zero-shot Groq baseline.

---

## Community

**r/soccer** — one of Reddit's largest sports communities, covering leagues, tournaments, players, tactics, and match results worldwide. The community is a strong fit for classification because a single thread contains tactically reasoned comments, match predictions, emotional reactions, and jokes — meaningful distinctions recognizable to regular members.

---

## Labels

| Label | Definition |
|---|---|
| **analysis** | Evaluates players, teams, tactics, or matches using reasoning, evidence, or football knowledge. Includes subjective judgments if supported by observation or argument. |
| **prediction** | Forecasts a future outcome or discusses a hypothetical scenario. Includes all forecasts, even if emotional or speculative. |
| **reaction** | Expresses pure emotion, excitement, or personal feeling with no claim being made. Also covers bare assertions about a player or team made without supporting reasoning. |
| **humor** | Primary purpose is humor, sarcasm, a joke, or meme reference. If it asserts a football outcome, it is NOT humor even if sarcastic. |

---

## Dataset

- **Source:** r/soccer match threads (Argentina/Algeria Messi hat trick thread, Norway vs Senegal Haaland goal clip thread, France/Mbappe thread, Portugal/Ronaldo thread)
- **Total labeled examples:** 288
- **Collection method:** Manual collection and annotation. Claude (claude.ai) was used to pre-label batches; every label was reviewed and corrected manually before being committed to the dataset.
- **Training dataset:** 160 examples (downsampled for balance — see below)

### Label Distribution (Full Dataset)

| Label | Count |
|---|---|
| analysis | 75 |
| reaction | 75 |
| humor | 75 |
| prediction | 63 |
| **Total** | **288** |

### Training Dataset (160 examples, used for fine-tuning)

| Label | Count |
|---|---|
| analysis | 40 |
| prediction | 40 |
| reaction | 40 |
| humor | 40 |

**Why downsampled:** The original dataset had Reaction at 80 examples (2x other labels). Training on the imbalanced set caused the model to collapse — predicting Reaction for every input regardless of content. Downsampling to 40 per label fixed this.

**Note on label merge:** An original 5-label taxonomy included a separate Opinion label (bare assertions without reasoning). This was merged into Reaction for model compatibility (the training notebook supported 4 labels). The Reaction definition was updated to cover both emotional responses and unsupported assertions.

### Difficult Labeling Examples

**Example 1:** *"mbappe might take the record this WC with how he's been scoring. Haaland probably won't be able to catch up."*
Possible: Prediction, Analysis → **Prediction.** The main point is a forecast; the scoring reference supports it rather than standing as independent analysis.

**Example 2:** *"It is ridiculous. But I do think tbf Senegal are a much better team than anyone Messi has played yet."*
Possible: Reaction, Analysis → **Analysis.** The comparative claim about opponent quality is reasoned judgment that could be fact-checked, not a bare assertion.

**Example 3:** *"Viking demands a blood sacrifice from France and Argentina"*
Possible: Humor, Reaction → **Humor.** The framing is entirely constructed for comedic effect; the literal reading makes no sense as a genuine response.

---

## Model

- **Base model:** `distilbert-base-uncased`
- **Task:** Sequence classification (4 labels)
- **Fine-tuning library:** HuggingFace Transformers + Trainer API

### Hyperparameter Decisions

| Parameter | Default | Final Value | Reason |
|---|---|---|---|
| `num_train_epochs` | 3 | 20 | 3 epochs caused majority-class collapse on 112 training examples. More passes needed; `load_best_model_at_end=True` prevents overfitting by saving the best checkpoint. |
| `learning_rate` | 2e-5 | 5e-5 | Higher learning rate helped the model escape the majority-class trap on a small dataset. |
| `per_device_train_batch_size` | 16 | 16 | Unchanged — fits T4 GPU comfortably. |

---

## Evaluation

### Test Set Performance

| Model | Accuracy | Macro F1 |
|---|---|---|
| Zero-shot baseline (Groq llama-3.3-70b-versatile) | 0.750 | 0.58 |
| Fine-tuned DistilBERT | 0.625 | 0.62 |

### Per-Class Metrics

**Zero-shot Baseline:**

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| analysis | 1.00 | 0.67 | 0.80 | 6 |
| prediction | 0.67 | 0.67 | 0.67 | 6 |
| reaction | 0.60 | 0.75 | 0.67 | 12 |
| humor | 0.20 | 0.17 | 0.18 | 6 |
| **accuracy** | | | **0.60** | **30** |

**Fine-tuned DistilBERT:**

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| analysis | 0.62 | 0.83 | 0.71 | 6 |
| prediction | 1.00 | 0.67 | 0.80 | 6 |
| reaction | 0.44 | 0.67 | 0.53 | 6 |
| humor | 0.67 | 0.33 | 0.44 | 6 |
| **accuracy** | | | **0.625** | **24** |

### Confusion Matrix

|  | Predicted: analysis | Predicted: prediction | Predicted: reaction | Predicted: humor |
|---|---|---|---|---|
| **True: analysis** | 5 | 0 | 1 | 0 |
| **True: prediction** | 0 | 4 | 2 | 0 |
| **True: reaction** | 1 | 0 | 4 | 1 |
| **True: humor** | 2 | 0 | 2 | 2 |

---

## Where Fine-Tuning Helped

Despite lower overall accuracy, fine-tuning produced meaningful improvements on two classes:

| Metric | Baseline | Fine-tuned | Change |
|---|---|---|---|
| Prediction F1 | 0.67 | 0.80 | **+0.13** |
| Humor F1 | 0.18 | 0.44 | **+0.26** |
| Macro F1 | 0.58 | 0.62 | **+0.04** |

Fine-tuning learned community-specific Humor patterns (dry sarcasm, meme references, absurdist jokes about Mbappe being a "dictator") that the zero-shot LLM consistently missed. Prediction also improved — the model learned the community's specific phrasing of forecasts, including emotionally-toned ones the LLM classified as Reaction.

---

## Wrong Predictions Analysis

**Wrong Prediction 1 — Reaction misclassified as Analysis (confidence: 0.93)**

> *"Don't care what anyone says. Messi is still the best in the world. He could still easily play in Europe if he wanted to. Would just need a hard working team around him like he has with Argentina."*

**True:** reaction | **Predicted:** analysis

The model was fooled by evaluative football vocabulary — "best in the world", "play in Europe", "hard working team" are all phrases it associates with Analysis. But no reasoning supports the claim; it's a pure assertion. This reveals the model learned surface vocabulary rather than the presence or absence of supporting argument.

---

**Wrong Prediction 2 — Humor misclassified as Reaction (confidence: 0.95)**

> *"My favorite part of when he joined Miami is seeing him go into grocery stores without people swarming him. Nobody knew who he was lol."*

**True:** humor | **Predicted:** reaction

The joke is implied — Messi being unknown at a grocery store is absurd given his global fame. But the comment uses personal conversational tone ("My favorite part") that reads like a genuine emotional response. The model never learned to detect implied absurdity as a humor signal, defaulting to Reaction for anything personal and expressive.

---

**Wrong Prediction 3 — Humor misclassified as Prediction (confidence: 0.84)**

> *"I've seen enough, Morocco is the 2025 AFCON champion"*

**True:** humor | **Predicted:** prediction

A documented edge case from planning.md — sarcastic outcome claims. The comment is clearly a joke (making a definitive tournament prediction based on one Senegal performance) but structurally looks exactly like a Prediction. The model correctly identified the future-outcome structure but missed the sarcastic intent entirely.

---
## Sample Classifications

| Text | True Label | Predicted | Confidence | Notes |
|---|---|---|---|---|
| "Saliba and Upamecano have not been seriously tested yet. They spend most of their game recovering loose balls and giving it back to their midfield." | analysis | analysis | 0.88 | ✅ Correct — tactical observation with specific positional detail, exactly what Analysis is designed to capture |
| "I think Haaland might get there. Needs to deliver against France though." | prediction | reaction | 0.44 | ❌ Short conditional forecast; model missed the future-outcome structure |
| "Before anyone feels bad for Algeria, the real victims are the Portugal players watching a certain teammate..." | humor | analysis | 0.96 | ❌ Evaluative vocabulary fooled the model into high-confidence wrong call |
| "We could see double digits for multiple players at this point wow" | prediction | reaction | 0.70 | ❌ Emotionally toned forecast classified as pure reaction |
| "0 goals in 16 matches for Bryne. Champions League and World Cup is all well and good, but clearly not cut out for the pressure of second tier Norwegian football." | humor | analysis | 0.95 | ❌ Stat-framed sarcasm — model read the numbers as Analysis evidence |

---

## Spec Reflection

**Where the spec helped:** The spec's emphasis on label design before any data collection was the most valuable constraint. Writing decision rules for edge cases (Analysis vs Reaction, Humor vs Prediction) before annotating 200 examples meant annotation was consistent rather than decided case-by-case. Without that upfront work the training signal would have been much noisier.

**Where implementation diverged:** The spec assumed fine-tuning would improve on the zero-shot baseline by at least 5 percentage points. The opposite happened — overall accuracy regressed from 0.750 to 0.625. This was partly a dataset size problem (112 training examples after splitting) and partly a label merge problem: collapsing Opinion into Reaction created a noisy class the model couldn't learn cleanly. The spec didn't anticipate that a 4-label constraint would force a taxonomy change that hurt one class's learnability.

---

## AI Tool Usage

**Instance 1 — Wrong prediction pattern analysis:** After running evaluation, I pasted all 9 misclassified examples into Claude and asked it to identify common patterns. It flagged that 5 of 9 errors involved Humor being misclassified as Analysis or Reaction, and that the Humor errors split into two subtypes: stat-framed sarcasm (model reads numbers as evidence → Analysis) and implied absurdity without explicit joke markers (model reads personal tone → Reaction). I verified both patterns against the confusion matrix — Humor had the lowest recall at 0.33 — and included them in the failure analysis.

**Instance 2 — Pre-labeling assistance:** Claude was used to suggest initial labels for batches of Reddit comments using the label definitions from planning.md. Every suggested label was reviewed manually before being committed to the dataset. Approximately 30% of suggestions were overridden, mostly on Humor/Reaction boundary cases where Claude defaulted to Reaction for dry sarcasm.

## What the Model Learned vs What Was Intended

**What it learned:**
- Vocabulary patterns associated with each class (football evaluation words → Analysis, future tense → Prediction)
- Community-specific Humor phrasing — jokes about Mbappe being a "dictator", Haaland as a lab-designed striker, absurdist record comparisons
- Prediction phrasing — emotionally-toned forecasts the LLM missed

**What it failed to learn:**
- The difference between a bare assertion and a supported claim (Analysis vs Reaction boundary)
- Implied absurdity and dry sarcasm without explicit joke markers (Humor vs Reaction)
- Context-dependent jokes that require knowing the community's meme vocabulary

**Root cause of regression on overall accuracy:**
The zero-shot LLM (70B parameters, deep football pretraining) has a strong prior on football discourse. Fine-tuning a 66M parameter model on 112 training examples cannot overcome that general knowledge advantage on Analysis and Reaction. The fine-tuned model won where community-specific patterns matter (Humor, Prediction) and lost where world knowledge matters (Analysis, Reaction).

---

## Reflection

The classifier did not meet the original success threshold of 75% accuracy and 5-point improvement over baseline (baseline: 0.750, fine-tuned: 0.625, regression: -0.125). However it revealed something more interesting: **fine-tuning a small model on limited community data produces a more balanced classifier** (macro F1 0.58 → 0.62) even when overall accuracy doesn't improve.

The Reaction label is the weakest point — merging Opinion and Reaction created a noisy class covering two conceptually different things. A future version with separate Opinion and Reaction labels and 200+ examples per class would likely improve significantly, especially with the 5-label taxonomy the project originally designed.

---

## Repo Structure

```
├── planning.md               # Label taxonomy, data plan, evaluation criteria
├── README.md                 # This file
├── takemeter_balanced.csv    # Training dataset (160 examples, 40 per label)
├── confusion_matrix.png      # Confusion matrix from fine-tuned model
└── evaluation_results.json   # Side-by-side baseline vs fine-tuned metrics
```

---
