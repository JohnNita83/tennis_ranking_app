#!/bin/bash

# Ensure we fail fast if anything goes wrong
set -e

# Set environment variables for local dev
export FLASK_APP=app.py
export FLASK_ENV=development   # enables debug mode + auto reload
export PATH="$HOME/.local/bin:$PATH"

# Start Flask (foreground so you see logs directly)
python -m flask run --host=0.0.0.0 --port=5000
