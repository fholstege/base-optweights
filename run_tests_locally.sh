#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

echo "Simulating GitHub Actions workflow locally"


# Step 1: Clean up any existing build artifacts
echo "Cleaning up old build artifacts..."
rm -rf build dist *.egg-info



# Step 2: Set up Python (assuming you have pyenv installed)
echo "Setting up Python..."
pyenv local 3.9.12  # This sets up all versions. Adjust as needed.

# Step 3: Create and activate a virtual environment
echo "Creating virtual environment..."
python -m venv .env.test
source .env.test/bin/activate


# Step 4: Install dependencies, including optweights
echo "Installing dependencies..."
pip install pytest
pip install torch
pip install -r requirements.txt
echo "Local installation of optweights..."
pip install -e .

# Add this after pip install -e .
pip list | grep optweights
python -c "import optweights; print(optweights.__file__)"

# Step 5: Run tests
echo "Running tests..."
PYTHONPATH=$PYTHONPATH:$(pwd) pytest tests/

# Deactivate virtual environment
deactivate
rm -rf .env.test

echo "Local simulation completed"