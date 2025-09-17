# AGN Classifier Research Assets

This repository contains the datasets, exploratory notebooks, and consolidated
Python modules that supported the research published in the [Journal of Student
Research](https://www.jsr.org/hs/index.php/path/article/view/6420).  The goal of
this refresh is to make the project easier to navigate and to provide a
repeatable workflow for generating model metrics from the SIMBAD cross-matched
catalogues used in the paper.

## Repository layout

```
├── data
│   └── raw/                 # Original Excel data exports used in the analysis
├── docs/                    # Manuscript and reference documents
├── notebooks/               # Historical notebooks (optimised for Google Colab)
├── reports/                 # Generated CSV reports from the modelling pipeline
└── src/
    ├── notebook_exports/    # Script exports of the notebooks for version control
    └── pipeline.py          # Reproducible data preparation & modelling workflow
```

The `src/pipeline.py` module turns the exploratory steps from the notebooks into
a single command-line workflow that produces descriptive statistics, model
metrics and feature importances for the eFEDS × VLASS × SIMBAD catalogue.

## Getting started

1. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\\Scripts\\activate`
   ```

2. Install the core dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the modelling pipeline:

   ```bash
   python src/pipeline.py
   ```

   The command will populate the `reports/` directory with three CSV files:

   - `efeds_feature_summary.csv` – descriptive statistics for the modelling
     features.
   - `efeds_feature_importances.csv` – feature importance scores from the random
     forest baseline.
   - `efeds_model_metrics.csv` – accuracy, precision and recall for the baseline
     models.

## Working with the notebooks

The notebooks inside the `notebooks/` folder reflect the original exploratory
workflow and expect a Google Colab environment with access to Google Sheets via
`gspread`.  For traceability the notebooks are also exported as Python scripts
under `src/notebook_exports/`.  These exports are not meant to replace the
notebooks, but they provide a version-controlled record of the analysis code.

## Data usage notes

- The `data/raw/` directory contains the unaltered Excel catalogues referenced
  in the publication.  Please preserve these files to ensure the pipeline and
  notebooks remain reproducible.
- Generated artefacts live under `reports/`.  Delete the CSV files before rerun
  if you prefer a clean output directory.

## Contributing

The repository now uses Black for code formatting.  Run `black src` before
committing to maintain a consistent style.  Issues and pull requests are
welcome if you build on top of the published research.
