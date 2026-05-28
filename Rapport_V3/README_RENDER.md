# Rapport_V3 rendering

From PowerShell:

```powershell
cd Rapport_V3
quarto render Rapport_V3.qmd --to pdf
```

The report expects figures in:

- `../results_v3/figures/main_test_accuracy.pdf`
- `../results_v3/figures/main_final_test_accuracy.pdf`
- `../results_v3/figures/gradient_cosine_similarity.pdf`
- `../results_v3/figures/class_accuracy_heatmaps.pdf`
- `../results_v3/figures/class_accuracy_curves.pdf`
- `../results_v3/figures/forgetting_by_epoch.pdf`
- `../results_v3/figures/lr_sweep_final_accuracy.pdf`
- `../results_v3/figures/lr_sweep_accuracy_curves.pdf`
- `../results_v3/figures/embedding_pca_grid.pdf`
