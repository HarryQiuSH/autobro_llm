# Code Quality Setup

This project uses Ruff for code style checking and formatting, replacing pylint for better performance and comprehensive linting.

## GitHub Actions

The project includes automated code quality checks that run on:
- Push to `main` and `develop` branches
- Pull requests to `main` and `develop` branches
- Manual workflow dispatch

### Workflow Jobs

1. **Ruff Linting and Formatting**: Checks code style and formatting compliance
2. **Type Checking**: Runs mypy for static type analysis (non-blocking initially)

## Local Development Setup

### Install Development Dependencies

```bash
# Using pip
pip install -e ".[dev]"

# Using uv (recommended)
uv sync --extra dev
```

### Pre-commit Hooks

Set up pre-commit hooks to catch issues before committing:

```bash
# Install pre-commit
pip install pre-commit

# Install the git hook scripts
pre-commit install

# (Optional) Run against all files
pre-commit run --all-files
```

## Manual Code Quality Checks

### Ruff Linting

```bash
# Check for linting issues
ruff check .

# Fix auto-fixable issues
ruff check --fix .

# Check specific files
ruff check app.py document_processor.py
```

### Ruff Formatting

```bash
# Check formatting
ruff format --check .

# Apply formatting
ruff format .

# Check specific files
ruff format --check app.py
```

### Type Checking

```bash
# Run mypy type checker
mypy --ignore-missing-imports .

# Check specific files
mypy app.py
```

## Configuration Files

- `.github/workflows/code-quality.yml`: GitHub Actions workflow
- `ruff.toml`: Ruff configuration with comprehensive rule set
- `.pre-commit-config.yaml`: Pre-commit hooks configuration
- `pyproject.toml`: Project configuration with development dependencies

## Ruff Rules Enabled

The configuration enables comprehensive linting including:
- **E, W**: pycodestyle errors and warnings
- **F**: pyflakes
- **I**: isort (import sorting)
- **N**: pep8-naming
- **UP**: pyupgrade
- **S**: flake8-bandit (security)
- **B**: flake8-bugbear
- **A**: flake8-builtins
- **C4**: flake8-comprehensions
- **SIM**: flake8-simplify
- **Q**: flake8-quotes
- **RET**: flake8-return
- **TCH**: flake8-type-checking

## Ignored Rules

Some rules are ignored for practical development:
- **S101**: Use of assert (allowed for testing)
- **T201**: Print statements (allowed)
- **B008**: Function calls in argument defaults
- **S603, S607**: Subprocess security warnings

## Integration with VS Code

Add these settings to your VS Code `settings.json`:

```json
{
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "none",
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
        }
    }
}
