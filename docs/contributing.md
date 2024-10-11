# Contributing to the Climate Risk Assessment Tool

We welcome contributions to the Climate Risk Assessment Tool! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment as described in [Getting Started](getting_started.md)

## Making Changes

1. Create a new branch for your feature or bug fix
2. Make your changes, adhering to the coding style (see below)
3. Write or update tests for your changes
4. Run the test suite to ensure all tests pass
5. Commit your changes with a clear and descriptive commit message

## Coding Style

- Follow PEP 8 guidelines for Python code
- Use type hints for function arguments and return values
- Write docstrings for all functions, classes, and modules
- Keep functions focused and small (aim for under 50 lines)
- Use meaningful variable and function names

## Testing

- Write unit tests for new functions and classes
- Update existing tests if you change functionality
- Aim for at least 80% test coverage for new code
- Run the full test suite before submitting a pull request:
  ```
  pytest
  ```

## Documentation

- Update the README.md file if you change functionality
- Add or update docstrings for any modified functions or classes
- If adding new features, update the relevant documentation files in the `docs/` directory

## Submitting Changes

1. Push your changes to your fork on GitHub
2. Create a pull request from your fork to the main repository
3. Describe your changes in the pull request, linking any relevant issues
4. Wait for a maintainer to review your pull request
5. Make any requested changes and update your pull request

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

## Questions?

If you have any questions about contributing, please open an issue or contact the maintainers.

Thank you for contributing to the Climate Risk Assessment Tool!