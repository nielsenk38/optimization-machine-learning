# Render instructions

From PowerShell:

```powershell
cd reportv2
quarto render reportv2.qmd --to pdf
```

The report expects figures at:

- `../results/figures/test_accuracy.pdf`
- `../results/figures/final_test_accuracy.pdf`
- `../results/figures/lr_sensitivity_label_orders.pdf`
- `../results/figures/prediction_distribution_label_orders.pdf`
