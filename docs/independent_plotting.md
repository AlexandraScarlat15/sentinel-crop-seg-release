# Independent U-Net plots from existing metrics (no training or inference)

The curated software has four separate workflows:

1. **Generate a dataset** with Sentinel-2/EuroCrops inputs, sampling settings and a manifest.
2. **Train a U-Net** only if desired, on an existing manifest; the trainer saves checkpoint files and training metrics.
3. **Evaluate or predict** using an existing trusted checkpoint, with no training.
4. **Plot prior results** from already saved CSV/JSON/NPY files, with no model loading, Copernicus access, or dataset generation.

No pretrained checkpoint is bundled with the current software-only draft.
The authors must approve the exact checkpoint and its model card before release.
These plotting commands can work before that approval when valid old reports
already exist.

## Training curves: standalone command

The training CLI writes \`training_metrics.csv\` and \`training_summary.json\`.
Plots can be made later (or on another computer) without reopening training:

\`\`\`bash
sentinel-crop-plot-training \
  --run-dir path/to/existing_training_run \
  --out-dir path/to/new_training_figures \
  --model-name "U-Net" --formats png pdf svg
\`\`\`

The results are three figures in selected formats:

- \`training_validation_loss\`: train/validation loss per epoch;
- \`validation_performance\`: validation mIoU and pixel accuracy in percent;
- \`learning_rate_schedule\`: per-epoch learning rate.

The best checkpoint epoch is highlighted from the original training summary.
Missing epochs, duplicates, invalid scores and a mismatch between saved
best validation mIoU and CSV history are rejected. **No training is started.**

## Evaluation figures: standalone command

First perform evaluation with \`sentinel-crop-evaluate-unet\` and a **trusted**
checkpoint, or use an existing complete evaluation directory. That directory
must contain \`confusion_matrix_counts.npy\` and \`summary.json\`.

\`\`\`bash
sentinel-crop-plot-evaluation \
  --eval-dir path/to/existing_evaluation \
  --out-dir path/to/new_evaluation_figures \
  --model-name "U-Net" --formats png pdf svg
\`\`\`

This generates confusion-matrix counts, ground-truth-row-normalized
percentages and per-class IoU bars in the selected formats. The plot
directory also contains \`metrics_per_class_pretty.csv\`,
\`confusion_matrix_counts.csv\`,
\`confusion_matrix_row_norm_percent.csv\` and
\`plots_summary.json\` recording the source report and class names.

For custom class taxonomies, create a JSON list **in class-ID order**:

\`\`\`json
["background", "wheat", "barley"]
\`\`\`

Then pass \`--class-names-json path/to/classes.json\`. If not specified, class
names come from the original \`metrics_per_class.csv\` (preferred); otherwise
the baseline 11-class taxonomy is used for exactly 11 classes, and generic
\`class_0\`, etc. for other class counts. The matrix is **not limited to
11 classes**. Class IDs, count shape and optional labels are validated.

Rows mean **ground truth**; columns mean **predictions**. Normalization
divides each row by its ground-truth support and displays **percent**.
A class absent from the ground truth has zero row-normalized values and
zero-valued undefined rates (these are not evidence of model quality).
The per-class table computes precision, recall, Dice/F1 and IoU from
the original pixel counts, not from rounded display values.

## Safety and provenance

- These commands only read existing metrics; they never load a checkpoint,
  make predictions, train a model, fetch imagery or change dataset splits.
- Plot output directories must **not already exist**; source metrics stay
  unchanged. Use different output folders for different experiments.
- Figures are diagnostic/presentation artifacts; they do not substitute for
  evaluating the actual approved model with its original preprocessing,
  source manifest, corrected-mask version and held-out test protocol.
- The original historical scripts under development \`tools/\` remain
  internal. These separate package commands are their portable user-facing
  counterparts and avoid manuscript-specific directory assumptions.

Validate from synthetic files only:

\`\`\`bash
python -m pytest tests/test_training_plots.py tests/test_evaluation_plots.py -q
\`\`\`
