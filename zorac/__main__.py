"""
Entry point for running Zorac as a Python module: `python -m zorac`

When Python encounters `python -m zorac`, it looks for a `__main__.py` file inside
the `zorac` package directory. This file is that entry point.

This is separate from the console script entry point defined in pyproject.toml
(which calls `zorac.main:main` directly). Both paths ultimately call the same
`main()` function, ensuring consistent behavior regardless of how the user
launches the application.

Why separate from main.py?
  - Python's `-m` flag requires `__main__.py` by convention.
  - Keeping it minimal (just an import and call) follows the principle of
    separation of concerns — the actual application logic lives in main.py.
  - The `if __name__ == "__main__"` guard ensures this code only runs when
    the file is executed directly, not when it's imported.
"""

from .main import main

if __name__ == "__main__":
    main()
