# Contributing to Advanced Fingerprint Matcher

We welcome contributions from the community! This document provides guidelines for contributing to the Advanced Fingerprint Matcher project.

## 🚀 Quick Start for Contributors

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/advance_fingermatcher.git
cd advance_fingermatcher

# Add upstream remote
git remote add upstream https://github.com/JJshome/advance_fingermatcher.git
```

### 2. Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all extras
pip install -e ".[dev,ml,viz]"

# Install pre-commit hooks
pre-commit install
```

### 3. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=advance_fingermatcher tests/

# Run specific test files
pytest tests/test_basic.py -v
```

## 🛠️ Development Guidelines

### Code Style

We use several tools to maintain code quality:

```bash
# Format code
black advance_fingermatcher/
black tests/

# Sort imports
isort advance_fingermatcher/
isort tests/

# Lint code
flake8 advance_fingermatcher/

# Type checking
mypy advance_fingermatcher/
```

### Code Standards

- **PEP 8**: Follow Python style guidelines
- **Type Hints**: Use type hints for function signatures
- **Docstrings**: Write comprehensive docstrings for all public functions
- **Tests**: Write tests for new functionality
- **Error Handling**: Include proper error handling and logging

### Example Code Style

```python
from typing import Optional, Tuple, List
import numpy as np
from pathlib import Path

def match_fingerprints(
    image1: np.ndarray,
    image2: np.ndarray,
    quality_threshold: float = 0.5,
    use_descriptors: bool = True
) -> Tuple[float, Optional[dict]]:
    """
    Match two fingerprint images using enhanced algorithms.
    
    Args:
        image1: First fingerprint image as numpy array
        image2: Second fingerprint image as numpy array  
        quality_threshold: Minimum quality threshold (0.0-1.0)
        use_descriptors: Whether to use rich descriptors
        
    Returns:
        Tuple of (match_score, detailed_results)
        
    Raises:
        ValueError: If images are invalid
        RuntimeError: If matching fails
    """
    if image1 is None or image2 is None:
        raise ValueError("Images cannot be None")
    
    try:
        # Implementation here
        match_score = 0.85
        results = {"matched_minutiae": 12, "total_minutiae": 15}
        return match_score, results
        
    except Exception as e:
        raise RuntimeError(f"Matching failed: {e}") from e
```

## 📝 Pull Request Process

### 1. Create Feature Branch

```bash
# Create and switch to feature branch
git checkout -b feature/your-feature-name

# Make your changes
# ...

# Commit changes
git add .
git commit -m "feat: add new matching algorithm"
```

### 2. Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(matching): add enhanced bozorth3 algorithm
fix(cli): resolve import error in demo command
docs: update installation instructions
test: add comprehensive matching tests
```

### 3. Submit Pull Request

```bash
# Push feature branch
git push origin feature/your-feature-name

# Create pull request on GitHub
# - Fill out the pull request template
# - Link to relevant issues
# - Add appropriate labels
```

### 4. PR Review Process

1. **Automated Checks**: CI/CD pipeline runs automatically
2. **Code Review**: Maintainers review the code
3. **Feedback**: Address any review comments
4. **Approval**: PR approved by maintainers
5. **Merge**: PR merged into main branch

## 🐛 Bug Reports

### Before Reporting

1. **Search existing issues** to avoid duplicates
2. **Try the latest version** to see if the bug is already fixed
3. **Minimal reproduction** - create the smallest possible example

### Bug Report Template

```markdown
## Bug Description
Brief description of the bug

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Expected Behavior
What you expected to happen

## Actual Behavior
What actually happened

## Environment
- OS: [e.g., Ubuntu 20.04, Windows 10, macOS 12]
- Python version: [e.g., 3.9.7]
- Package version: [e.g., 1.0.1]
- Dependencies: [any relevant package versions]

## Additional Context
Any other relevant information
```

## 💡 Feature Requests

### Before Requesting

1. **Check existing features** - maybe it already exists
2. **Search existing requests** to avoid duplicates
3. **Consider alternatives** - is there another way to achieve your goal?

### Feature Request Template

```markdown
## Feature Description
Clear description of the proposed feature

## Use Case
Why is this feature needed? What problem does it solve?

## Proposed Solution
How do you envision this working?

## Alternatives Considered
Other approaches you considered

## Additional Context
Any other relevant information
```

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── test_basic.py              # Basic functionality tests
├── test_matcher.py            # Matching algorithm tests
├── test_minutiae_detector.py  # Detection tests
├── test_batch_processor.py    # Batch processing tests
├── data/                      # Test data files
└── fixtures/                  # Test fixtures
```

### Writing Tests

```python
import pytest
from advance_fingermatcher import AdvancedFingerprintMatcher

class TestAdvancedMatcher:
    """Test cases for AdvancedFingerprintMatcher."""
    
    @pytest.fixture
    def matcher(self):
        """Create matcher instance for testing."""
        return AdvancedFingerprintMatcher()
    
    def test_basic_matching(self, matcher):
        """Test basic fingerprint matching."""
        # Arrange
        image1 = self.create_test_image()
        image2 = self.create_test_image()
        
        # Act
        score = matcher.match_images(image1, image2)
        
        # Assert
        assert 0.0 <= score <= 1.0
        assert isinstance(score, float)
    
    def test_invalid_input(self, matcher):
        """Test handling of invalid input."""
        with pytest.raises(ValueError):
            matcher.match_images(None, None)
```

### Test Categories

1. **Unit Tests**: Test individual functions/classes
2. **Integration Tests**: Test component interactions
3. **Performance Tests**: Test speed and memory usage
4. **End-to-End Tests**: Test complete workflows

## 📚 Documentation

### Code Documentation

- **Docstrings**: All public functions must have docstrings
- **Type Hints**: Use type hints for better code clarity
- **Comments**: Explain complex algorithms and business logic

### External Documentation

```
docs/
├── getting_started.md      # Installation and basic usage
├── api_reference.md        # Complete API documentation
├── algorithms/             # Algorithm explanations
├── tutorials/              # Step-by-step guides
└── examples/               # Code examples
```

### Documentation Style

- **Clear and Concise**: Easy to understand
- **Examples**: Include code examples
- **Up-to-Date**: Keep in sync with code changes
- **Comprehensive**: Cover all features

## 🏷️ Issue Labels

We use labels to categorize issues and PRs:

### Type Labels
- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to docs
- `question`: Further information is requested

### Priority Labels
- `priority:high`: Critical issues
- `priority:medium`: Important issues
- `priority:low`: Nice-to-have improvements

### Status Labels
- `status:ready`: Ready for development
- `status:in-progress`: Currently being worked on
- `status:blocked`: Blocked by external factors

## 🎯 Areas for Contribution

### High-Priority Areas

1. **Algorithm Improvements**
   - Enhanced matching algorithms
   - Performance optimizations
   - Accuracy improvements

2. **Testing & Quality**
   - Comprehensive test coverage
   - Performance benchmarks
   - Error handling improvements

3. **Documentation**
   - Tutorial improvements
   - API documentation
   - Code examples

4. **Integration**
   - Database connectors
   - Cloud platform support
   - Third-party integrations

### Good First Issues

Look for issues labeled `good first issue` - these are beginner-friendly:

- Documentation improvements
- Simple bug fixes
- Adding tests
- Code cleanup tasks

## 🤝 Community Guidelines

### Code of Conduct

- **Be Respectful**: Treat everyone with respect
- **Be Inclusive**: Welcome newcomers and diverse perspectives
- **Be Constructive**: Provide helpful feedback
- **Be Professional**: Maintain professional communication

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Pull Requests**: Code contributions and reviews

## 🏆 Recognition

Contributors will be recognized in:

- **CONTRIBUTORS.md**: List of all contributors
- **Release Notes**: Acknowledgment of significant contributions
- **README.md**: Special recognition for major contributors

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Advanced Fingerprint Matcher! 🔍✨
