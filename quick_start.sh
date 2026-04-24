#!/bin/bash

# Quick Start Script for Live Interaction Mode

echo "=========================================="
echo "Live Interaction Mode - Quick Start"
echo "=========================================="
echo ""

# Check for required environment variables
echo "Checking environment variables..."

if [ -z "$AWS_ACCESS_KEY_ID" ]; then
    echo "⚠️  AWS_ACCESS_KEY_ID not set"
    echo "   Set it with: export AWS_ACCESS_KEY_ID='your_key'"
fi

if [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "⚠️  AWS_SECRET_ACCESS_KEY not set"
    echo "   Set it with: export AWS_SECRET_ACCESS_KEY='your_secret'"
fi

echo ""
echo "Available example configurations:"
echo "  1. configs/example_basic.json       - Basic conversation (Bedrock mode)"
echo "  2. configs/example_with_tools.json  - Conversation with tools"
echo "  3. configs/example_scripted.json    - Scripted messages"
echo ""

echo "Running example_basic.json..."
echo ""

python main.py --config configs/example_basic.json

echo ""
echo "=========================================="
echo "Session Complete!"
echo "Check logs/ directory for results"
echo "=========================================="
