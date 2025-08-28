#!/bin/bash

# ==============================
# Setup Script for Jetson Project
# ==============================

# Fail fast
set -e

echo "🚀 Starting setup for Jetson..."

# 1. Update system
echo "📦 Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# 2. Install Python & system dependencies
echo "🐍 Installing Python3, pip, and venv..."
sudo apt-get install -y python3 python3-pip python3-venv git

# Install PyQt5 if using GUI (adjust if PySide6 instead)
echo "🖼 Installing PyQt5 (for GUI)..."
sudo apt-get install -y python3-pyqt5

# 3. Setup virtual environment
if [ ! -d "venv" ]; then
    echo "📂 Creating virtual environment..."
    python3 -m venv venv
fi

echo "🔗 Activating virtual environment..."
source venv/bin/activate

# 4. Install Python dependencies
if [ -f "requirements.txt" ]; then
    echo "📚 Installing requirements..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "⚠️ No requirements.txt found!"
fi

# 5. Prepare output folder
if [ ! -d "output" ]; then
    echo "📂 Creating output directory..."
    mkdir output
fi

# 6. Check ONNX model
MODEL_PATH="app/core/models/EfficientNetb0100_baseline.onnx"
if [ ! -f "$MODEL_PATH" ]; then
    echo "⚠️ Model file not found at $MODEL_PATH"
    echo "👉 Please place the ONNX model manually."
else
    echo "✅ ONNX model found: $MODEL_PATH"
fi

echo "🎉 Setup completed successfully!"
echo "👉 To start, run: source venv/bin/activate && python3 app/gui/main_gui.py"
