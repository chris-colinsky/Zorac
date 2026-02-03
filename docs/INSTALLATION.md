# Installation Guide

This guide covers all methods for installing Zorac.

## Homebrew (macOS/Linux - Recommended)

The easiest way to install Zorac on macOS or Linux:

```bash
# Add the tap
brew tap chris-colinsky/zorac

# Install
brew install zorac

# Verify installation
zorac --help
```

### Upgrading

```bash
brew upgrade zorac
```

### Uninstalling

```bash
brew uninstall zorac
```

## pipx (All Platforms - Recommended)

[pipx](https://pipx.pypa.io/) installs Python applications in isolated environments, preventing dependency conflicts:

```bash
# Install pipx if you don't have it
brew install pipx       # macOS/Linux
# or: pip install pipx  # Any platform

# Install zorac
pipx install zorac

# Verify installation
zorac --help
```

### Upgrading

```bash
pipx upgrade zorac
```

### Uninstalling

```bash
pipx uninstall zorac
```

## pip (All Platforms)

Standard Python package installation:

```bash
# Install
pip install zorac

# Verify installation
zorac --help
```

### Upgrading

```bash
pip install --upgrade zorac
```

### Uninstalling

```bash
pip uninstall zorac
```

## uv (For Developers)

Using the modern [uv](https://github.com/astral-sh/uv) package manager:

```bash
# Install globally
uv tool install zorac

# Or run without installing (ephemeral)
uvx zorac

# Verify installation
zorac --help
```

### Upgrading

```bash
uv tool upgrade zorac
```

### Uninstalling

```bash
uv tool uninstall zorac
```

## From Source (Development)

For contributing or development:

```bash
# Clone the repository
git clone https://github.com/chris-colinsky/zorac.git
cd zorac

# Install dependencies
uv sync

# Run from source
uv run zorac

# Or activate venv and run directly
source .venv/bin/activate
zorac
```

See [Development Guide](DEVELOPMENT.md) for more details on contributing.

## Windows Users

Zorac runs on Windows via [WSL (Windows Subsystem for Linux)](https://learn.microsoft.com/en-us/windows/wsl/install):

1. Install WSL: `wsl --install`
2. Open your Linux distribution
3. Follow the pip or pipx installation instructions above

## Verification

After installation, verify Zorac is working:

```bash
zorac --help
```

On first run, Zorac will guide you through configuration.

## Data Directory

Zorac stores all user data in `~/.zorac/`:

| File | Purpose |
|------|---------|
| `config.json` | Server URL, model, and settings |
| `session.json` | Conversation history |
| `history` | Command-line history |

To completely remove Zorac and its data:

```bash
# Uninstall (use your installation method)
pipx uninstall zorac   # or pip, brew, etc.

# Remove data
rm -rf ~/.zorac
```

## Next Steps

After installation:

- [Configuration Guide](CONFIGURATION.md) - Configure your vLLM server and settings
- [Usage Guide](USAGE.md) - Learn Zorac commands and features
- [Server Setup](SERVER_SETUP.md) - Set up your vLLM inference server
