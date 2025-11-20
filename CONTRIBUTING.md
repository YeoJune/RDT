# Contributing to RDT

Thank you for your interest in contributing to RDT!

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/yourusername/rdt.git
cd rdt
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install in Development Mode

```bash
pip install -e ".[dev]"
```

This installs:
- RDT package in editable mode
- All dependencies
- Development tools (pytest, black, flake8, isort)

### 4. Verify Installation

```bash
python check_compatibility.py
python test_model.py
```

## Code Style

We follow PEP 8 with some modifications:

### Formatting with Black

```bash
# Format all code
black rdt/

# Check without modifying
black --check rdt/
```

Configuration in `pyproject.toml`:
- Line length: 100
- Target Python: 3.8+

### Import Sorting with isort

```bash
# Sort imports
isort rdt/

# Check without modifying
isort --check rdt/
```

### Linting with flake8

```bash
flake8 rdt/
```

## Testing

### Run Tests

```bash
# Run model tests
python test_model.py

# Run pytest (if test suite exists)
pytest tests/

# Run with coverage
pytest --cov=rdt tests/
```

### Writing Tests

Tests should go in `tests/` directory:

```python
# tests/test_model.py
import torch
from rdt import RDT

def test_rdt_forward():
    model = RDT(vocab_size=1000, d_model=128)
    x = torch.randint(0, 1000, (2, 10))
    hidden, logits, gate = model(x)
    assert logits.shape == (2, 10, 1000)
```

## Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write code following the style guide
- Add tests for new features
- Update documentation

### 3. Run Checks

```bash
# Format code
black rdt/
isort rdt/

# Lint
flake8 rdt/

# Test
python test_model.py
pytest tests/
```

### 4. Commit

```bash
git add .
git commit -m "Add feature: description"
```

Commit message format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `style:` Formatting
- `refactor:` Code restructuring
- `test:` Tests
- `chore:` Maintenance

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Code Organization

### Package Structure

```
rdt/
├── __init__.py          # Package exports
├── model.py            # Model architecture
├── data.py             # Data loading
├── trainer.py          # Training logic
├── utils.py            # Utilities
├── configs/            # Config files
└── scripts/            # CLI scripts
```

### Adding New Features

#### New Model Component

Add to `rdt/model.py`:
```python
class NewComponent(nn.Module):
    """Description"""
    def __init__(self, ...):
        ...
    
    def forward(self, x):
        ...
```

Export in `rdt/__init__.py`:
```python
from .model import NewComponent
__all__ = [..., "NewComponent"]
```

#### New Configuration

Add to `rdt/configs/`:
```yaml
# new_config.yaml
model:
  ...
training:
  ...
```

Update docs in `USAGE_GUIDE.md`.

#### New Script

Add to `rdt/scripts/`:
```python
# new_script.py
def main():
    ...

if __name__ == '__main__':
    main()
```

Add entry point in `setup.py`:
```python
entry_points={
    'console_scripts': [
        'rdt-new=rdt.scripts.new_script:main',
    ],
}
```

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def function(arg1: int, arg2: str) -> bool:
    """One-line summary.
    
    Longer description if needed.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
    
    Returns:
        Description of return value
    
    Example:
        >>> result = function(1, "test")
        >>> print(result)
        True
    """
    ...
```

### Update Documentation

When adding features:
1. Update docstrings
2. Update README.md (if user-facing)
3. Update USAGE_GUIDE.md (for detailed usage)
4. Update INSTALL.md (for installation changes)

## Release Process

### 1. Update Version

In `rdt/__init__.py` and `setup.py`:
```python
__version__ = "0.2.0"
```

In `pyproject.toml`:
```toml
version = "0.2.0"
```

### 2. Update Changelog

Create or update `CHANGELOG.md`:
```markdown
## [0.2.0] - 2024-XX-XX

### Added
- New feature X
- New feature Y

### Fixed
- Bug fix Z

### Changed
- Improvement W
```

### 3. Build Package

```bash
./build.sh
```

### 4. Test Installation

```bash
pip install dist/rdt_transformer-*.whl
python check_compatibility.py
```

### 5. Create Git Tag

```bash
git tag -a v0.2.0 -m "Release version 0.2.0"
git push origin v0.2.0
```

### 6. Upload to PyPI (Maintainers Only)

```bash
pip install twine
twine upload dist/*
```

## Issue Guidelines

### Reporting Bugs

Include:
- Python and package versions
- Operating system
- Full error traceback
- Minimal code to reproduce

### Requesting Features

Include:
- Use case description
- Proposed API/interface
- Example usage

### Asking Questions

- Check existing issues first
- Provide context
- Show what you've tried

## Code Review

Pull requests will be reviewed for:
- Code quality and style
- Test coverage
- Documentation
- Breaking changes

Expect feedback and be ready to make changes.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to:
- Open an issue
- Start a discussion
- Contact maintainers

Thank you for contributing to RDT!
