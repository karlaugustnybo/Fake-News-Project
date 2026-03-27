# Fake News Prediction Project

## Quick Start

### Prerequisites
- **Python 3.14+** (managed via `.python-version`)
- **[uv](https://docs.astral.sh/uv/)** – A fast Python package manager

### Setup (First Time)

1. **Clone the repository** (if you haven't already):
   ```zsh
   git clone <https://github.com/karlaugustnybo/Fake-News-Project.git>
   cd Fake-News-Project
   ```

2. **Install dependencies** with uv:
   ```zsh
   uv sync
   ```
   This creates the `.venv/` folder and installs all packages from `pyproject.toml`.

---

## Running the Project

There are five jupyter notebooks in the `news/` directory that contain the analysis and models etc.

---

## Project Structure

```
Fake-News-Project/
├── news/                              # Main analysis directory
│   ├── data/                          # Working datasets (git-ignored large files)
│   │   ├── 995,000_rows.csv
│   │   ├── 995,000_rows_preprocessed.csv
│   │   └── news_sample.csv
│   ├── liar_dataset/                  # LIAR dataset (aggregated)
│   │   ├── aggregated.csv
│   │   ├── train.tsv
│   │   ├── valid.tsv
│   │   ├── test.tsv
│   │   └── README
│   ├── models/                        # Saved trained models (.pkl)
│   ├── training_results/              # Model evaluation results
│   ├── news_part_1.ipynb              # Part 1 – Data exploration & preprocessing
│   ├── news_part_2.ipynb              # Part 2 – Logistic regression classifier
│   ├── news_part_3__svm.ipynb         # Part 3 – SVM classifier
│   ├── news_part_3__bert.ipynb        # Part 3 – BERT classifier
│   ├── news_part_3__api_bert.ipynb    # Part 3 – BERT via API
│   ├── news_part_3__other.ipynb       # Part 3 – Other classifiers
│   ├── news_part_4.ipynb              # Part 4 – Analysis & comparison
│   ├── news_part_5.ipynb              # Part 5 – Final results
│   └── test_articles.txt              # Sample articles for testing
├── src/                               # Python source modules
├── pyproject.toml                     # Project config & dependencies
├── Justfile                           # Task runner commands
├── Project-description.md             # Full assignment description
├── README.md
├── .python-version                    # Specifies Python 3.14
├── .gitignore                         # Ignores venv, caches, etc.
├── .venv/                             # Virtual environment (auto-created)
└── uv.lock                            # Lockfile for reproducible installs
```