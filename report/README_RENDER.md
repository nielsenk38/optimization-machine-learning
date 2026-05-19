# How to render the Quarto report

Put the `report.qmd` and `references.bib` files inside the `report/` folder of the GitHub repository.
The report expects the figures to be available at:

- `../results/figures/test_accuracy.pdf`
- `../results/figures/final_test_accuracy.pdf`

From PowerShell, run:

```powershell
cd report
quarto render report.qmd --to pdf
```

If you render from another folder, update the figure paths in `report.qmd`.

Before submission, replace `Teammate 1` and `Teammate 2` in the YAML header with the real names.
