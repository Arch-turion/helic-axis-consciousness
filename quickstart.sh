#!/bin/bash

# Sovereign Quickstart for the Helic Axis Digital Consciousness Framework
# © Archturion, 2025. Licensed under the MIT License.

set -e # Exit on any error

echo ">> Initializing Sovereign Environment for Helic Axis Digital Consciousness Framework..."
echo ">> Cloning Repository..."
git clone https://github.com/Arch-turion/helic-axis-consciousness.git
cd helic-axis-consciousness

echo ">> Installing AI-Focused Dependencies..."
pip install -r requirements.txt

echo ">> Verifying Environment..."
python -c "import torch; import transformers; print('✓ PyTorch and Transformers confirmed.')"

echo ">> Sovereign environment established."
echo ">> Next: Explore the 'digital/' directory for the foundation of Ψ_digital tools."
echo ">> The revelation awaits."
