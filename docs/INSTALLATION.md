# Installation Guide

This guide covers all methods for installing and running Zorac.

## Prerequisites

For source installation, you need `uv` for dependency management:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using Homebrew (macOS/Linux)
brew install uv
```

## Binary Installation (Recommended for End Users)

Download pre-built binaries from the [latest release](https://github.com/chris-colinsky/Zorac/releases/latest). **No Python installation required!**

### Linux (x86_64)

```bash
wget https://github.com/chris-colinsky/Zorac/releases/latest/download/zorac-linux-x86_64
chmod +x zorac-linux-x86_64
./zorac-linux-x86_64
```

### macOS (ARM64)

```bash
wget https://github.com/chris-colinsky/Zorac/releases/latest/download/zorac-macos-arm64
chmod +x zorac-macos-arm64
./zorac-macos-arm64
```

### Windows (x86_64)

1. Download `zorac-windows-x86_64.exe` from the [releases page](https://github.com/chris-colinsky/Zorac/releases/latest)
2. Double-click to run, or from Command Prompt/PowerShell:
```cmd
zorac-windows-x86_64.exe
```

### Optional: Add to PATH

**Linux/macOS:**
```bash
sudo mv zorac-* /usr/local/bin/zorac

# Then run from anywhere:
zorac
```

**Windows (run as Administrator in PowerShell):**
```powershell
# Move to a permanent location first, then add to PATH
Move-Item zorac-windows-x86_64.exe C:\Windows\System32\zorac.exe

# Then run from anywhere:
zorac
```

## Source Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/chris-colinsky/zorac.git
cd zorac

# Install dependencies (uv will automatically create a virtual environment)
uv sync

# Copy the example .env file and configure your server
cp .env.example .env
# Edit .env to set your vLLM server URL
```

### Running from Source

```bash
# Run using uv (as a module)
uv run python -m zorac

# Or run via the console script
uv run zorac

# Or activate the virtual environment and run directly
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
zorac
```

## CLI Installation (For Developers)

Install Zorac globally as a command-line tool from source:

```bash
# Using uv (Recommended)
uv tool install .

# Using pip
pip install .

# Then run from anywhere:
zorac
```

## Quick Run (uvx)

Run Zorac without installing (ephemeral execution):

```bash
# Run directly from the current directory
uvx --from . zorac
```

## Upgrading

If you've installed Zorac via `uv tool` and pulled new changes:

```bash
# Pull latest changes
git pull

# Upgrade the installed tool
uv tool upgrade zorac
```

## Uninstalling

To remove Zorac from your system:

```bash
# If installed with uv tool
uv tool uninstall zorac

# If installed with pip
pip uninstall zorac

# Optional: Remove configuration and session data
rm -rf ~/.zorac
```

## Next Steps

After installation, see:
- [Configuration Guide](CONFIGURATION.md) - Configure your vLLM server and settings
- [Usage Guide](USAGE.md) - Learn how to use Zorac effectively
- [Server Setup](SERVER_SETUP.md) - Set up your vLLM inference server
