.PHONY: help install install-dev test test-verbose test-coverage lint format type-check clean pre-commit run all-checks

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Zorac - Self-Hosted Local LLM Chat Client"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	uv sync

install-dev: ## Install development dependencies
	uv sync --extra dev

test: ## Run tests
	uv run pytest

test-verbose: ## Run tests with verbose output
	uv run pytest -vv

test-coverage: ## Run tests with coverage report
	uv run pytest --cov=zorac --cov-report=term-missing --cov-report=html
	@echo ""
	@echo "Coverage report generated in htmlcov/index.html"

test-watch: ## Run tests in watch mode (requires pytest-watch)
	uv run ptw

lint: ## Run linter (ruff)
	uv run ruff check .

lint-fix: ## Run linter and auto-fix issues
	uv run ruff check --fix .

format: ## Format code with ruff
	uv run ruff format .

format-check: ## Check code formatting without changing files
	uv run ruff format --check .

type-check: ## Run type checker (mypy)
	uv run mypy zorac.py

pre-commit-install: ## Install pre-commit hooks
	uv run pre-commit install

pre-commit: ## Run pre-commit hooks on all files
	uv run pre-commit run --all-files

clean: ## Remove generated files and caches
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete

run: ## Run the application
	uv run zorac.py

all-checks: lint type-check test ## Run all checks (lint, type-check, test)
	@echo ""
	@echo "✅ All checks passed!"

ci: install-dev all-checks ## Run CI pipeline (install deps + all checks)
	@echo ""
	@echo "✅ CI pipeline completed successfully!"

coverage-html: test-coverage ## Open coverage report in browser
	@if command -v open > /dev/null; then \
		open htmlcov/index.html; \
	elif command -v xdg-open > /dev/null; then \
		xdg-open htmlcov/index.html; \
	else \
		echo "Coverage report available at htmlcov/index.html"; \
	fi

dev-setup: install-dev pre-commit-install ## Complete development setup
	@echo ""
	@echo "✅ Development environment setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Copy .env.example to .env and configure your vLLM server"
	@echo "  2. Run 'make test' to verify everything works"
	@echo "  3. Run 'make run' to start the chat client"

stats: ## Show project statistics
	@echo "Project Statistics:"
	@echo "==================="
	@echo -n "Lines of code (zorac.py): "
	@wc -l zorac.py | awk '{print $$1}'
	@echo -n "Lines of tests: "
	@wc -l tests/*.py | tail -1 | awk '{print $$1}'
	@echo -n "Test files: "
	@ls tests/test_*.py | wc -l
	@echo -n "Python version: "
	@cat .python-version
