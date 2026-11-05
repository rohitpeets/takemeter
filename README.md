# TakeMeter

A fine-tuned text classifier for comments in r/soccer, one of Reddit's largest sports communities. It sorts comments into Analysis, Prediction, Reaction and Humor, and compares a fine-tuned DistilBERT model with a zero-shot LLM baseline.

A single r/soccer match thread mixes tactical breakdowns, predictions, emotional reactions and jokes, and regulars can tell them apart. That makes it a good test of whether a small model can learn a community's distinctions.

## Labels

| Label | Definition |
|---|---|
| analysis | Evaluates players, teams, tactics or matches with reasoning, evidence or football knowledge. Subjective judgments count if they're backed by observation or argument. |
| prediction | Forecasts a future outcome or discusses a hypothetical, including emotional or speculative forecasts. |
| reaction | Pure emotion or personal feeling with no claim, plus bare assertions about a player or team with no supporting reasoning. |
| humor | Mainly a joke, sarcasm or meme reference. A comment that asserts a football outcome is not humor, even if sarcastic. |

The original taxonomy had a fifth label, Opinion, for bare assertions. It was merged into Reaction because the training notebook supported four labels.

## Dataset

- Source: r/soccer match threads (Argentina/Algeria with a Messi hat trick, Norway vs Senegal with a Haaland goal clip, France/Mbappe, Portugal/Ronaldo).
- 288 labeled comments: 75 analysis, 75 reaction, 75 humor, 63 prediction.
- Collected and labeled by hand. Claude pre-labeled batches from the definitions in `planning.md`; I reviewed every label and overrode about 30%, mostly Humor/Reaction cases where it defaulted to Reaction for dry sarcasm.
- The original fine-tuning set was downsampled to 160 (40 per label). With Reaction at twice the size of the other classes, the model collapsed into predicting Reaction for everything.

Hard labeling calls:

- *"mbappe might take the record this WC with how he's been scoring. Haaland probably won't be able to catch up."* Prediction, not Analysis: the main point is a forecast, and the scoring reference supports it.
- *"It is ridiculous. But I do think tbf Senegal are a much better team than anyone Messi has played yet."* Analysis, not Reaction: the comparison of opponent quality is a reasoned, checkable judgment.
- *"Viking demands a blood sacrifice from France and Argentina"* Humor, not Reaction: it only makes sense as a joke.

## Model

`distilbert-base-uncased` (66M parameters) fine-tuned for 4-way sequence classification with HuggingFace Transformers and the Trainer API.

| Parameter | Default | Used | Why |
|---|---|---|---|
| `num_train_epochs` | 3 | 20 | 3 epochs collapsed to the majority class on the small training set. `load_best_model_at_end=True` keeps the best checkpoint. |
| `learning_rate` | 2e-5 | 5e-5 | Helped the model get out of the majority-class trap. |
| `per_device_train_batch_size` | 16 | 16 | Fits a T4 GPU. |

## Results: 5-fold cross-validation (Oct 2026)

5-fold stratified cross-validation over all 288 comments (`takemeter_300.csv`, seed 42). The run used a CRLF copy (sha256 `a769168b…d174c`) with the same 288 rows as this repo's LF copy (sha256 `b6f12d0b…de222`). In every fold both models are scored on the same held-out comments (57 to 58 per fold), so every comment is scored exactly once. DistilBERT keeps the hyperparameters above, trains on all training-fold rows without downsampling, and picks its checkpoint on a validation split from the training fold.

Groq retired `llama-3.3-70b-versatile` on 2026-08-16, so the zero-shot baseline is `openai/gpt-oss-120b` on Groq (reasoning effort low). All 288 baseline replies parsed.

| Model | Macro F1 (mean ± std) | Accuracy (mean ± std) | Pooled macro F1 [95% CI] |
|---|---|---|---|
| Fine-tuned DistilBERT (66M) | 0.542 ± 0.043 | 0.549 ± 0.048 | 0.545 [0.488, 0.600] |
| Zero-shot gpt-oss-120b | 0.518 ± 0.039 | 0.524 ± 0.036 | 0.521 [0.458, 0.579] |

The pooled macro F1 difference is +0.024, with a paired bootstrap 95% CI of [−0.049, +0.099]. The CI includes zero, so the two models are statistically indistinguishable on this data. Fine-tuning scored higher in 3 of 5 folds.

Per-class F1, pooled over all 288 comments:

| Label | Fine-tuned | Baseline |
|---|---|---|
| analysis | 0.70 | 0.51 |
| prediction | 0.56 | 0.65 |
| reaction | 0.46 | 0.53 |
| humor | 0.46 | 0.38 |

The fine-tuned model leads on Analysis and Humor; the baseline leads on Prediction and Reaction.

Raw outputs: `cv/takemeter_cv_results.json` (summary, CIs, per-class F1) and `cv/takemeter_cv_per_fold.csv`.

## Original single split (superseded)

The first evaluation scored the fine-tuned model on 24 held-out comments and the 70B baseline on a different 30, so the two were never compared on the same comments. It reported accuracy 0.625 and macro F1 0.62 for DistilBERT versus 0.750 and 0.58 for the baseline, but the baseline's own per-class table works out to 18/30 = 0.60 accuracy. Its per-class pattern (fine-tuning helping Prediction and Humor) rested on 6 to 12 examples per class and did not hold up in cross-validation. The numbers are kept in `evaluation_results.json` and `confusion_matrix.png`.

## Error analysis

These examples come from the original split, but the patterns are the useful part.

- *"Don't care what anyone says. Messi is still the best in the world. He could still easily play in Europe if he wanted to..."* True: reaction. Predicted: analysis (0.93). Football vocabulary like "best in the world" and "play in Europe" pulled it toward Analysis, though nothing supports the claim. The model learned surface vocabulary, not whether an argument is present.
- *"My favorite part of when he joined Miami is seeing him go into grocery stores without people swarming him. Nobody knew who he was lol."* True: humor. Predicted: reaction (0.95). The joke is implied absurdity, and the personal tone read as a genuine reaction.
- *"I've seen enough, Morocco is the 2025 AFCON champion"* True: humor. Predicted: prediction (0.84). Sarcastic outcome claims look exactly like predictions; `planning.md` flagged this edge case in advance.
- *"0 goals in 16 matches for Bryne. Champions League and World Cup is all well and good, but clearly not cut out for the pressure of second tier Norwegian football."* True: humor. Predicted: analysis (0.95). The model read the stat as evidence instead of setup for sarcasm.

What the model didn't learn: the difference between a bare assertion and a supported claim (Analysis vs Reaction), and dry sarcasm or implied absurdity without explicit joke markers (Humor vs Reaction and Prediction).

## Spec reflection

The spec required label definitions and edge-case rules before any data collection. Deciding Analysis vs Reaction and Humor vs Prediction up front kept annotation consistent across 288 comments instead of being decided case by case.

The spec expected fine-tuning to beat the baseline by at least 5 points. It didn't: in cross-validation the gap is +2.4 macro F1 points and not significant. Two causes stand out: the training set is small, and merging Opinion into Reaction created a noisy class that's hard to learn. A version with separate Opinion and Reaction labels and 200+ examples per class is the obvious next step.

## AI usage

- Pre-labeling: Claude suggested labels from the definitions in `planning.md`; I reviewed all of them and overrode about 30%.
- Error patterns: I gave Claude the misclassified examples from the original split and asked for common patterns. It found that 5 of 9 errors were Humor misread as Analysis or Reaction, split between stat-framed sarcasm and implied absurdity. I checked both against the confusion matrix (Humor recall was 0.33) before using them in the analysis.

## Repo structure

```
├── planning.md               # Label taxonomy, data plan, evaluation criteria
├── README.md
├── takemeter_balanced.csv    # Original training set (160 examples, 40 per label)
├── confusion_matrix.png      # Confusion matrix from the original split
├── evaluation_results.json   # Original single-split metrics (superseded)
└── cv/                       # 5-fold cross-validation outputs (Oct 2026)
```
