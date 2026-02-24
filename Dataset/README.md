# Preloaded Datasets Folder

This folder contains all the **preloaded datasets stored in tensor format** that are used by the application.

Instead of generating datasets repeatedly during runtime, the application loads these pre-saved tensors directly when required. This approach significantly improves performance and ensures consistency across experiments.

## Purpose

The datasets in this folder are used for:

* Faster application startup and training
* Avoiding repeated dataset generation
* Maintaining reproducible experiments
* Reducing computational overhead
* Providing ready-to-use training data for users

## Format

All datasets are stored in **tensor format** (e.g., PyTorch `.pt` or `.pth` files), making them efficient to load directly into the model training pipeline.

## How It Works

When a user selects a dataset inside the application:

1. The app checks this folder.
2. Loads the corresponding tensor file.
3. Uses it directly for training and visualization.

This eliminates the need to regenerate datasets each time the application runs.

## Important Notes

* Do not rename files unless you update the corresponding file paths in the code.
* Ensure tensor shapes and formats remain consistent with the expected model input.
* If you add new datasets, follow the same naming and storage format.

## Benefits

* Faster execution
* Better reproducibility
* Cleaner workflow
* Reduced computational cost

---

If you plan to add new datasets, simply save them in tensor format and place them in this directory so the application can load them when needed.
