# Brain Tumor Type Classification from Gene Expression Data

A machine learning pipeline that classifies brain tumor subtypes (ependymoma, glioblastoma, medulloblastoma, pilocytic astrocytoma, and healthy tissue) from microarray gene expression data. Tackling a classic high-dimensional, low-sample-size (HDLSS) problem common in genomics.

## Overview

This project uses the **CuMiDa Brain Cancer Gene Expression dataset** (GSE50161): 130 tissue samples, each measured across 54,675 genes. With ~420x more features than samples, the core challenge isn't building a model, it's compressing the feature space without losing the biological signal, then making the model's predictions interpretable.

The pipeline covers:
- Stratified train/test splitting for a small, imbalanced multi-class dataset
- Dimensionality reduction via PCA (54,675 genes → 77 components, retaining 95% of variance)
- Classification with a Random Forest
- Model interpretability using SHAP, tracing predictions back to individual genes

## Dataset

- **Source:** [CuMiDa — Brain Cancer Gene Expression, GSE50161](https://sbcb.inf.ufrgs.br/cumida)
- **Samples:** 130
- **Features:** 54,675 gene expression probes
- **Classes:** ependymoma, glioblastoma, medulloblastoma, pilocytic astrocytoma, normal (healthy tissue)

## Approach

1. **Data inspection** - checked shape, types, and missing values
2. **Target exploration** - visualized class distribution to flag imbalance up front
3. **Preprocessing** - dropped the sample ID column, label-encoded the target, standardized features
4. **Stratified split** - 104 train / 26 test, preserving class proportions given the small sample size
5. **PCA** - reduced 54,675 genes to 77 principal components (95% variance retained)
6. **Random Forest classifier** - trained on the PCA-reduced features
7. **Cross-validation** - 5-fold CV to get a more reliable performance estimate than a single split
8. **Evaluation** - classification report and confusion matrix on the held-out test set
9. **Explainability (SHAP)** - trained a second Random Forest directly on the top 500 most variable genes (skipping PCA, so features stay biologically identifiable) and used SHAP to find which genes drive each prediction

## Results

**Cross-validation accuracy:** 0.856 ± 0.073

**Test set performance:**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| ependymoma | 0.73 | 0.89 | 0.80 | 9 |
| glioblastoma | 1.00 | 0.86 | 0.92 | 7 |
| medulloblastoma | 1.00 | 0.50 | 0.67 | 4 |
| normal | 1.00 | 1.00 | 1.00 | 3 |
| pilocytic astrocytoma | 0.75 | 1.00 | 0.86 | 3 |

**Overall accuracy:** 85% (macro F1: 0.85, weighted F1: 0.84)

The model performs strongly overall, with perfect classification of healthy tissue and glioblastoma precision. The main weak point is medulloblastoma recall (0.50). The confusion matrix shows 2 of 4 medulloblastoma samples misclassified as ependymoma, likely reflecting both overlapping expression signatures and the very small sample count for this class (only 4 test samples).

**Interpretable model (top 500 genes, no PCA) accuracy:** 96.2%, notably higher than the PCA-based model, suggesting the top-variance genes alone carry most of the discriminative signal.

**Top genes driving predictions (via SHAP):**

Probe IDs were mapped to gene symbols using [bioDBnet](https://biodbnet-abcc.ncifcrf.gov/db/db2db.php), then cross-referenced against GeneCards, OMIM, and primary literature for known function.

| Probe ID | Gene | Mean \|SHAP\| | Known Function |
|---|---|---|---|
| 220156_at | CLXN | 0.0096 | Calcium-binding protein required for outer dynein arm assembly in motile cilia; mutations cause primary ciliary dyskinesia |
| 205932_s_at | MSX1 | 0.0088 | Homeobox transcription factor with a documented dual role as an oncogene or tumor suppressor across several cancers, including interaction with the p53 pathway |
| 1562371_s_at | VWA3B | 0.0073 | Brain-expressed protein linked to hereditary cerebellar ataxia (SCAR22) |
| 1553734_at | AK7 | 0.0061 | Maintains motile cilia structure and beat frequency; loss is linked to hydrocephalus and ciliary dyskinesia |
| 242162_at | DAW1 | 0.0056 | Assembly factor for dynein motor complexes in motile cilia; mutations cause primary ciliary dyskinesia |
| 229170_s_at | CFAP70 | 0.0053 | Regulates ciliary motility; specifically localizes to ependymal cilia lining the brain's ventricles |
| 231077_at | CFAP126 | 0.0049 | Member of the cilia/flagella-associated protein family, involved in axonemal ciliary structure |
| 204465_s_at | INA | 0.0048 | Neuronal intermediate filament; clinically used biomarker for 1p/19q-codeleted oligodendroglial gliomas |
| 220591_s_at | EFHC2 | 0.0048 | Calcium-binding domain protein implicated in brain-related conditions; family member (EFHC1) functions specifically at neuronal cilia |
| 239916_at | CFAP52 | 0.0044 | Cilia/flagella-associated protein functionally linked to motile cilia in the ependyma |

**Biological interpretation:** Seven of the ten top-ranked genes (CLXN, AK7, DAW1, CFAP70, CFAP126, CFAP52, and related family member EFHC2) are structural or motility components of motile cilia. This is a striking and biologically sensible pattern: ependymoma — the largest class in this dataset (35% of samples) — arises from ependymal cells, which are ciliated cells lining the brain's ventricles. It's plausible the model is partly separating ependymoma from other tumor types by picking up on residual ciliary gene expression, a signature specific to the ependymal cell of origin. INA adds a second, independently well-documented angle: it's an established clinical biomarker for distinguishing oligodendroglial tumors, relevant to differentiating glioma subtypes.

## Tech Stack

- Python, pandas, NumPy
- scikit-learn (PCA, Random Forest, cross-validation, metrics)
- SHAP (model explainability)
- matplotlib, seaborn (visualization)

## Project Structure

```
├── brain-tumor-type-classification.ipynb   # Full analysis notebook
├── README.md
└── data/
    └── Brain_GSE50161.csv                  # Not included — download from source below
```

## How to Run

1. Download the dataset from [CuMiDa](https://sbcb.inf.ufrgs.br/cumida) or [Kaggle](https://www.kaggle.com/datasets/brunogrisci/brain-cancer-gene-expression-cumida)
2. Install dependencies: `pip install pandas numpy scikit-learn shap matplotlib seaborn`
3. Run the notebook cells in order

## Limitations

- Very small dataset (130 samples) means test-set metrics, especially for minority classes like medulloblastoma, should be read with caution
- 5-fold cross-validation on an already-small training set means some folds contain very few examples of rarer classes
- The ciliary-gene pattern is a plausible, literature-supported hypothesis rather than a confirmed causal mechanism — it would need validation against a larger, independent dataset to draw firm biological conclusions

## Future Work

- Validate the ciliary-gene hypothesis with GO/pathway enrichment analysis (e.g., via `gseapy`) across the full top-500 gene set, not just the top 10
- Try alternative dimensionality reduction (e.g., feature selection via ANOVA F-test) for comparison against PCA
- Experiment with class-weighting or SMOTE to address medulloblastoma's low recall

## Data Source

Grisci, B. et al. CuMiDa: Curated Microarray Database. Original data: [GSE50161](http://sbcb.inf.ufrgs.br/carbm/static/cumida/Genes/Brain/GSE50161/Brain_GSE50161.csv)
