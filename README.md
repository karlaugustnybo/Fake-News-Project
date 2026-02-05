# Fake News Prediction Project

## 📋 Quick Start

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

## 🚀 Running the Project

Use `just` to run common commands:

| Command            | What it does                                       |
|--------------------|---------------------------------------------------|
| `just run`         | Runs `main.py` after type-checking and formatting |
| `just notebook`    | Opens an interactive [Marimo](https://marimo.io/) notebook |
| `just check`       | Type-check the codebase with [ty](https://docs.astral.sh/ty/) |
| `just format`      | Lint & format code with [Ruff](https://docs.astral.sh/ruff/) |
| `just test`        | Run tests with pytest                             |
| `just clean`       | Delete cache folders                              |

---

## 📂 Project Structure

```
Fake-News-Project/
├── main.py                 # Entry point – start coding here!
├── pyproject.toml          # Project config & dependencies
├── Justfile                # Task runner commands
├── Project-description.md  # Full assignment description
├── README.md               # This file
├── .python-version         # Specifies Python 3.14
├── .gitignore              # Ignores venv, caches, etc.
├── .venv/                  # Virtual environment (auto-created)
└── uv.lock                 # Lockfile for reproducible installs
```

---

## 🛠 Key Technologies

| Tool / Library     | Purpose                                      |
|-------------------|---------------------------------------------|
| **Polars**        | Fast DataFrame library (like Pandas)        |
| **Altair**        | Declarative data visualization              |
| **scikit-learn**  | Machine learning models                     |
| **PyTorch**       | Deep learning framework                     |
| **Marimo**        | Reactive Python notebooks                   |
| **DuckDB**        | In-process SQL database                     |
| **Pydantic AI**   | AI integrations (optional)                  |
| **WandB**         | Experiment tracking                         |
| **Rich**          | Pretty terminal output                      |

---

## 🔧 Development Tools

- **Ruff** – Linting and formatting
- **ty** – Type checking
- **pytest** – Testing framework

Run `just format` before committing to keep the code clean!

---

## 📖 Assignment Details

Read `Project-description.md` for the full project requirements, deadlines, and deliverables.

---

## 👥 Team Workflow

1. **Pull latest changes** before starting work:
   ```zsh
   git pull
   ```

2. **Format & test** before pushing:
   ```zsh
   just format
   just check
   just test
   ```

3. **Push**
   ```zsh
   git add .
   git commit -m "Your commit message"
   git push
   ```