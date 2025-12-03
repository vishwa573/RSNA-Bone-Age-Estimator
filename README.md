# PRML Bone Age Prediction Project

End-to-end implementation of a **bone age prediction** system for the RSNA Bone Age dataset, designed to satisfy the Pattern Recognition and Machine Learning (PRML) course project requirements:

- **Task 1 – Regression:** predict bone age as a **continuous value in years** from a hand X‑ray.
- **Task 2 – Classification:** predict an **ordinal developmental stage** (Child / Adolescent / Adult) discretized from the continuous bone age.
- **Analysis:** report standard metrics for both tasks, plus **gender-wise bias analysis** and visual explanations (Grad‑CAM).

---

## 1. Dataset & Folder Layout

**Expected dataset (Kaggle RSNA Bone Age):**

- `boneage-training-dataset.csv`
- `boneage-training-dataset/` (folder containing training X‑rays as `<id>.png`)
- Optionally: `boneage-test-dataset.csv` and `boneage-test-dataset/` (competition test set, unused by default)

Place these under:

```text
PRML_Bone_Age_Project/
  data/
    raw/
      boneage-training-dataset.csv
      boneage-training-dataset/
        boneage-training-dataset/
          10000.png
          10001.png
          ...
      boneage-test-dataset.csv           # optional
      boneage-test-dataset/              # optional
        boneage-test-dataset/
          4360.png
          ...
    processed/                            # will contain pre-resized images (optional)
```

The CSV must contain at least the columns described in the course spec:

- `id` – integer image identifier (no extension).
- `bone_age` – bone age in **months** (continuous target before conversion).
- `sex` or `male` – biological sex (`M`/`F` or binary flag).

> The code automatically converts `bone_age` from **months to years** and creates a discretized **stage** label.

---

## 2. Implemented Learning Tasks

### 2.1 Regression – Bone Age in Years

- Mapping: \( f_\text{reg}: \text{Image} \to \text{Age (years)} \).
- In `src/dataset.py`, we create:
  - `age_years = bone_age / 12.0` – this is the **regression target**.
- Loss:
  - `nn.L1Loss()` (MAE-style) for robustness.
- Metrics (in `src/evaluate.py`):
  - **MAE** – Mean Absolute Error.
  - **RMSE** – Root Mean Squared Error (computed from MSE for compatibility with older scikit‑learn).
  - **R² score** – coefficient of determination.

### 2.2 Classification – Developmental Stage

- Mapping: \( f_\text{cls}: \text{Image} \to \{\text{Child},\text{Adolescent},\text{Adult}\} \).
- Stages are derived in `src/dataset.py` from `age_years`:
  - `0` – **Child**: `< 12` years.
  - `1` – **Adolescent**: `12–18` years.
  - `2` – **Adult**: `> 18` years.
- Loss:
  - `nn.CrossEntropyLoss()`.
- Metrics (in `src/evaluate.py`):
  - **Accuracy**.
  - **Weighted F1-score** (handles class imbalance).
  - **Quadratic Weighted Kappa (QWK)** – captures agreement with ordinal structure.
  - Confusion matrix (values) and plotted confusion matrix (via `src/utils.py`).

Both tasks are trained **jointly** in a single multi-task network.

---

## 3. Model Architecture

File: `src/model.py`

- Backbone: **ImageNet-pretrained ResNet18** (or ResNet50, configurable).
  - Final FC layer is replaced by `nn.Identity()` to obtain a feature vector.
- Two task-specific heads:
  - `reg_head`: `Linear(in_features, 1)` → predicted age in years.
  - `cls_head`: `Linear(in_features, 3)` → logits for 3 developmental stages.
- Forward output:
  - `age_pred` – shape `(B,)`.
  - `stage_logits` – shape `(B, 3)`.

This shares low-level features while learning both regression and classification signals.

---

## 4. Data Loading, Preprocessing, and Splits

### 4.1 Dataset class & transforms

File: `src/dataset.py`

- `BoneAgeDataset`:
  - Reads the main CSV.
  - Normalizes column names.
  - Derives:
    - `age_years` from `bone_age / 12.0`.
    - `stage` (0/1/2) from `age_years` (Child / Adolescent / Adult).
    - `sex_int` from `sex` or `male` column (1 = male, 0 = female).
  - Filters out rows where the corresponding image file is missing.
  - Returns per-sample dict:
    - `image`: transformed RGB tensor.
    - `age_years`: float32.
    - `stage`: long int label.
    - `sex`: long int (0/1).

- `get_transforms(img_size, train=True)`:
  - Always: `Resize((IMG_SIZE, IMG_SIZE))`, `ToTensor()`, `Normalize(ImageNet mean/std)`.
  - During training: `RandomHorizontalFlip`, `RandomRotation` for augmentation.

### 4.2 Preprocessing script (optional, one-time)

File: `preprocess.py`

- Reads the CSV from `data/raw`.
- Locates raw images under Kaggle’s nested folder:
  - `data/raw/boneage-training-dataset/boneage-training-dataset/`.
- Resizes images to `Config.IMG_SIZE` × `Config.IMG_SIZE`.
- Saves them as `<id>.png` in `data/processed/`.

This lets you avoid repeated resizing cost during training.

### 4.3 Train / validation / test split

Implemented in `create_dataloaders` inside `main.py`:

- Reads full dataset from the CSV.
- Uses **scikit-learn** `train_test_split` to create:
  - 70% **train**.
  - 15% **validation**.
  - 15% **test**.
- Stratifies splits by `stage` to preserve class distribution.
- Creates separate `DataLoader`s for train/val/test with appropriate transforms.

---

## 5. Training & Evaluation

### 5.1 CLI entry point

File: `main.py`

Supported modes:

- `--mode train` – Train the multi-task model.
- `--mode evaluate` – Load best checkpoint and evaluate on test set + generate plots.

Key optional arguments:

- `--epochs` – number of training epochs (default 10).
- `--batch_size` – batch size (default 32).
- `--lr` – learning rate (default 1e-4).
- `--use_processed` – if set, load images from `data/processed` instead of `data/raw`.

### 5.2 Trainer logic

File: `src/train.py`

- Uses `Trainer` class with:
  - Losses: `L1Loss` (regression) + `CrossEntropyLoss` (classification).
  - Optimizer: `Adam` with configurable LR and weight decay.
  - Total loss: `total_loss = reg_loss + cls_loss`.
  - Gradient clipping to `max_norm=5.0`.
- For each epoch:
  - Iterates over **train** loader, logs running total/reg/cls loss using `tqdm`.
  - Evaluates on **validation** set via `evaluate(split="val")`:
    - Prints losses + full set of regression and classification metrics.
  - Tracks best model (lowest validation total loss) and saves checkpoint:
    - `output/models/best_model.pt`.
- After training (if a test loader is provided):
  - Reloads best model.
  - Evaluates on **test** set via `evaluate(split="test")` with full metrics + bias.

### 5.3 Metrics and gender-wise bias

File: `src/evaluate.py`

- `regression_metrics` → MAE, RMSE, R².
- `classification_metrics` → accuracy, weighted F1, QWK.
- `gender_bias_regression` → MAE for males, MAE for females, difference.
- `gender_bias_classification` → accuracy for males, accuracy for females, difference.

`Trainer.evaluate` prints all of these for the given split.

---

## 6. Visualization & Explainability

File: `src/utils.py`

- `plot_scatter(true, pred, save_path=...)`:
  - Scatter plot of predicted vs true **ages in years** with y = x reference line.
- `plot_confusion_matrix(true, pred, class_names, save_path=...)`:
  - Heatmap of confusion matrix for the 3 developmental stages.
- `visualize_gradcam(model, input_tensor, target_layer, class_idx=None)`:
  - Lightweight **Grad‑CAM** implementation that returns a normalized heatmap over the input image.

File: `gradcam_demo.py`

- Loads the best model and a handful of test samples.
- Computes Grad‑CAM for each.
- Overlays heatmap on original X‑ray.
- Saves figures as `output/plots/gradcam_example_*.png`.

---

## 7. How to Run the Project

### 7.1 Setup

```bash
# From project root
python -m venv venv
venv\Scripts\Activate.ps1   # PowerShell on Windows
pip install -r requirements.txt

# (Recommended) ensure numpy < 2 for sklearn/torchvision compatibility
pip install "numpy<2.0" --upgrade
```

Place the **RSNA training CSV and images** as described in Section 1.

### 7.2 Optional: one-time preprocessing

```bash
python preprocess.py
```

This will create pre-resized images in `data/processed/`.

### 7.3 Train

Using **raw** images:

```bash
python main.py --mode train
```

Using **preprocessed** images:

```bash
python main.py --mode train --use_processed
```

You can override hyperparameters, e.g.:

```bash
python main.py --mode train --use_processed --epochs 20 --batch_size 16 --lr 1e-4
```

### 7.4 Evaluate & generate plots

```bash
python main.py --mode evaluate --use_processed
```

Outputs:

- `output/plots/scatter_pred_vs_true_test.png` – regression scatter.
- `output/plots/confusion_matrix_test.png` – classification confusion matrix.

### 7.5 Grad‑CAM visualizations

```bash
python gradcam_demo.py
```

Outputs:

- `output/plots/gradcam_example_1.png`, `gradcam_example_2.png`, ...

---

## 8. What This Project Delivers (per Course Spec)

- **Regression task** predicting bone age in **years** with MAE, RMSE, and R² metrics.
- **Classification task** predicting 3 developmental stages (`Child`, `Adolescent`, `Adult`) with accuracy, F1-score, QWK, and confusion matrix.
- **Joint multi-task CNN** with a shared pretrained ResNet backbone and two heads.
- **Train/val/test split** (70/15/15) with stratification by stage.
- **Gender-wise bias analysis** for both regression and classification.
- **Visualizations**: regression scatter plots, confusion matrix heatmaps, and Grad‑CAM heatmaps.
- A clean, reproducible **Python codebase** usable from the command line.

These features align with the PRML course project requirements for bone age prediction, including model design, metrics, bias analysis, and interpretability.
