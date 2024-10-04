#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

echo "Simulating GitHub Actions workflow locally"

# Step 1: Set up Python (assuming you have pyenv installed)
echo "Setting up Python..."
pyenv local 3.9.12  # This sets up all versions. Adjust as needed.

# Step 2: Create and activate a virtual environment
echo "Creating virtual environment..."
python -m venv venv
source venv/bin/activate

# step 2.5 Set up PYTHONPATH
export PYTHONPATH=src

# Step 3: Install dependencies
echo "Installing dependencies..."
python -m pip install --upgrade pip
pip install pytest
pip install -e .

# Step 4: Run tests
echo "Running tests..."
pytest tests/

# Deactivate virtual environment
deactivate

echo "Local simulation completed"