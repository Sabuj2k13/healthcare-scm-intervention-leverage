[README.md](https://github.com/user-attachments/files/35081249/README.md)
# Intervention leverage for healthcare supply chain risk networks

Data, code and decision-support tool for the manuscript:

> **From mapping to mitigation: A closed-form intervention leverage measure for healthcare supply chain risk networks** (submitted to *Healthcare Analytics*).

The study maps the cause–effect structure among sixteen healthcare supply chain risk factors in Bangladesh with Pythagorean fuzzy DEMATEL. It then ranks mitigation interventions with a closed-form leverage measure, ΔΣᵢ(λ) = λ Dᵢ(1 + Rᵢ) / (1 + λ tᵢᵢ), and finds optimal and budget-constrained intervention bundles by complete enumeration of all 2¹⁶ = 65,536 bundles.

## Contents

| Path | Description |
|---|---|
| `data/aggregated_direct_relation_matrix_A.csv` | Aggregated direct-relation matrix *A* (expected values averaged over ten experts) |
| `data/total_relation_matrix_T.csv` | Total-relation matrix *T* = *X*(*I* − *X*)⁻¹ |
| `data/factor_scores_and_leverage.csv` | Dispatched (*D*) and received (*R*) influence, prominence, relation and single-factor leverage (λ = 0.5) |
| `code/leverage_analysis.py` | Reproduces the leverage, bundle-enumeration, break-even, λ-sensitivity, submodularity, decision-robustness and model-validation analyses (Supplementary file S2) |
| `tool/decision_support_tool.html` | Self-contained interactive tool: open in any browser, enter your own intervention costs and budget (Supplementary file S3) |
| `supplementary/S4_Supplementary_Tables_and_Figure.docx` | Supplementary Tables S1–S4 and Supplementary Fig. S1 |
| `figures/` | Figures 1–6 and Fig. S1 of the manuscript |

The factor codes F1–F16 are defined in Table 1 of the manuscript. F13 is pandemic-related disruption impact; F14 is sudden demand shifts.

## Reproducing the results

Requires Python 3.9+.

```bash
pip install -r requirements.txt
python code/leverage_analysis.py                 # leverage, bundles, break-even ratios, λ sensitivity
python code/leverage_analysis.py --validation    # Section 5.7: closed-form verification + cascade simulation (~1 min)
python code/leverage_analysis.py --robustness    # submodularity + decision robustness (~15 min)
```

## Licence

Code: MIT licence (`LICENSE`). Data, figures and supplementary documents: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Citation

If you use these materials, please cite the article (citation details will be added on publication) or this repository (`CITATION.cff`).
