#!/bin/bash

# AI Debate Arena Setup Script with uv

set -e

echo "🏛️ Setting up AI Debate Arena..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✅ uv found"

# Create virtual environment and install dependencies
echo "📦 Installing dependencies with uv..."
uv sync

echo "🔧 Setting up environment..."

# Create .env file template if it doesn't exist
if [ ! -f .env ]; then
    cat > .env << EOF
# AI Debate Arena Configuration
# Copy this file and add your API keys

# Choose your LLM provider (uncomment one set):

# OpenAI (recommended)
# OPENAI_API_KEY=your-openai-api-key-here
# JUDGE_MODEL=gpt-4o-mini
# ARGUING_MODEL=gpt-4o-mini

# Anthropic
# ANTHROPIC_API_KEY=your-anthropic-api-key-here
# JUDGE_MODEL=claude-3-haiku-20240307
# ARGUING_MODEL=claude-3-haiku-20240307

# OpenRouter
# OPENROUTER_API_KEY=your-openrouter-api-key-here
# JUDGE_MODEL=openrouter/anthropic/claude-3-haiku
# ARGUING_MODEL=openrouter/openai/gpt-4o-mini

# Optional: Customize generation parameters
# TEMPERATURE=0.7
# JUDGE_TEMPERATURE=0.1
# MAX_NEW_TOKENS=300
# MAX_ARGUMENT_LENGTH=300
EOF
    echo "📝 Created .env template file"
    echo "   Please edit .env and add your API keys"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your API keys"
echo "2. Run the demo:"
echo "   uv run python app.py"
echo ""
echo "Or for development:"
echo "   uv shell"
echo "   python app.py"
echo ""