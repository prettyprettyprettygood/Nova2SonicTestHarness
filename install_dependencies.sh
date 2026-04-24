#!/bin/bash

# Installation script — creates .venv and installs all dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "Installing Dependencies"
echo "========================================"
echo ""

# Detect Python
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "❌ Python not found. Please install Python 3.12+"
    exit 1
fi

echo "Using: $PYTHON ($($PYTHON --version))"
echo ""

# Create venv if it doesn't exist
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv "$SCRIPT_DIR/.venv"
    echo "✅ .venv created"
else
    echo "✅ .venv already exists"
fi

# Activate venv
source "$SCRIPT_DIR/.venv/bin/activate"
echo "✅ Activated .venv ($(python --version))"
echo ""

# Upgrade pip
pip install --upgrade pip --quiet

# Install requirements
echo "Installing dependencies from requirements.txt..."
pip install -r "$SCRIPT_DIR/requirements.txt"
echo ""

# Verify critical imports
echo "========================================"
echo "Verifying installation..."
echo "========================================"

python -c "
import sys
packages = {
    'boto3': 'AWS SDK',
    'rx': 'Reactive streams (RxPY)',
    'pandas': 'Results management',
    'yaml': 'Config files',
    'dotenv': 'Environment loading',
}

missing = []
for pkg, desc in packages.items():
    try:
        __import__(pkg)
        print(f'  ✅ {pkg:20s} - {desc}')
    except ImportError:
        print(f'  ❌ {pkg:20s} - {desc}')
        missing.append(pkg)

# Check Nova Sonic SDK (experimental)
for pkg, desc in [
    ('aws_sdk_bedrock_runtime', 'Nova Sonic bidirectional streaming'),
    ('smithy_aws_core', 'AWS signing for streaming'),
]:
    try:
        __import__(pkg)
        print(f'  ✅ {pkg:20s} - {desc}')
    except ImportError:
        print(f'  ⚠️  {pkg:20s} - {desc} (install from Nova Sonic sample code)')
        missing.append(pkg)

if missing:
    print()
    print(f'{len(missing)} package(s) missing — see above')
    sys.exit(1)
else:
    print()
    print('All packages installed!')
"

echo ""
echo "========================================"
echo "✅ Done!"
echo ""
echo "To activate the venv in your terminal:"
echo "  source .venv/bin/activate"
echo ""
echo "Then configure credentials in .env and run:"
echo "  python main.py --config configs/example_basic.json"
echo "========================================"
