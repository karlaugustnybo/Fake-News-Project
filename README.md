# Fake News Prediction Project

## Quick Start

### Prerequisites
- **Python 3.14+** (managed via `.python-version`)
- **[uv](https://docs.astral.sh/uv/)** – A fast Python package manager
- **[just](https://github.com/casey/just)** – A command runner (like `make`)

### Setup (First Time)

1. **Clone the repository** (if you haven't already):
   ```zsh
   git clone <repo-url>
   cd Fake-News-Project
   ```

2. **Install dependencies** with uv:
   ```zsh
   uv sync
   ```
   This creates the `.venv/` folder and installs all packages from `pyproject.toml`.

---

## Running the Project

Use `just` to run common commands:

| Command            | What it does                                       |
|--------------------|---------------------------------------------------|
| `just check`       | Type-check the codebase with [ty](https://docs.astral.sh/ty/) |
| `just format`      | Lint & format code with [Ruff](https://docs.astral.sh/ruff/) |
| `just test`        | Run tests with pytest                             |
| `just clean`       | Delete cache folders                              |

---

## Project Structure

```
Fake-News-Project/
├── news/                        # Main analysis directory
│   ├── data/                    # Datasets (git-ignored large files)
│   │   ├── 995,000_rows.csv
│   │   ├── 995,000_rows_preprocessed.csv
│   │   └── news_sample.csv
│   ├── news_1_pl.ipynb          
│   ├── news_part_1__2.ipynb     
│   ├── news_part_2__1.ipynb     
│   ├── news_part_3.ipynb        
│   └── bert__1.ipynb            
├── pyproject.toml               # Project config & dependencies
├── Justfile                     # Task runner commands
├── Project-description.md       # Full assignment description
├── README.md                    
├── .python-version              # Specifies Python 3.14
├── .gitignore                   # Ignores venv, caches, etc.
├── .venv/                       # Virtual environment (auto-created)
└── uv.lock                      # Lockfile for reproducible installs
```


---

## Development Tools

- **Ruff** – Linting and formatting
- **ty** – Type checking
- **pytest** – Testing framework

Run `just format` before committing to keep the code clean!

---

## Assignment Details

Read `Project-description.md` for the full project requirements, deadlines, and deliverables.