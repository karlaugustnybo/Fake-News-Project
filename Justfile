run:
    just check
    just format
    uv run python main.py

notebook:
    uv run marimo edit --watch notebook.py

check:
    uv run ty check .

format:
    uv run ruff check --fix .
    uv run ruff format .

test:
    uv run pytest

clean:
    rm -rf __pycache__ .pytest_cache .ruff_cache

push:
    git add .
    git commit -m "Update"
    git push

pull:
    git pull

reset:
    git reset --hard origin/main
