#!/bin/bash

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Running training script..."
python3 train.py
