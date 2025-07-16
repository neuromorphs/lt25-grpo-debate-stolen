# 🏛️ AI Debate Arena - Demo

An interactive debate game where humans compete against AI on fun topics! Perfect for kids, educators, and anyone who loves a good argument.

## 🌟 Features

- **50+ Engaging Topics**: From "Video games should be taught as a school sport" to "All buildings should have rooftop gardens"
- **Real-time AI Debates**: Argue against sophisticated AI models on the same side of issues
- **Smart Judging**: AI judges evaluate arguments based on logic, structure, examples, tone, and counterarguments
- **Live Leaderboard**: Track your wins, losses, and format scores against other players and AI baseline
- **Multiple LLM Providers**: Support for OpenAI, Anthropic, OpenRouter, and local HuggingFace models
- **Format Coaching**: Real-time feedback on argument structure with XML formatting guidance

## 🚀 Quick Start

### 1. Install Dependencies

**With uv (recommended):**
```bash
cd demo
./setup.sh
```

**Or with pip:**
```bash
cd demo
pip install -r requirements.txt
```

### 2. Set Up API Keys

Choose your preferred LLM provider and set the appropriate environment variable:

**For OpenAI (recommended):**
```bash
export OPENAI_API_KEY="your-openai-api-key"
export JUDGE_MODEL="gpt-4o-mini"
export ARGUING_MODEL="gpt-4o-mini"
```

**For Anthropic:**
```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export JUDGE_MODEL="claude-3-haiku-20240307"
export ARGUING_MODEL="claude-3-haiku-20240307"
```

**For OpenRouter:**
```bash
export OPENROUTER_API_KEY="your-openrouter-api-key"
export JUDGE_MODEL="openrouter/anthropic/claude-3-haiku"
export ARGUING_MODEL="openrouter/openai/gpt-4o-mini"
```

### 3. Run the Demo

**With uv:**
```bash
uv run python app.py
```

**Or activate the virtual environment:**
```bash
uv shell
python app.py
```

The interface will be available at `http://localhost:7860`

## 🎮 How to Play

1. **Enter Your Name**: Join the leaderboard by entering your name
2. **Get a Topic**: Click "New Debate Topic" to receive a random topic and stance (PRO or CON)
3. **Write Your Argument**: Format your argument using the XML structure:
   ```xml
   <reasoning>
   Your step-by-step thinking process here...
   </reasoning>
   <answer>
   Your final 2-3 sentence argument here
   </answer>
   ```
4. **Submit & Compete**: The AI will generate a competing argument for the same stance
5. **Get Judged**: An AI judge evaluates both arguments and declares a winner
6. **Climb the Leaderboard**: Win debates to improve your ranking!

## ⚙️ Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JUDGE_MODEL` | `gpt-4o-mini` | Model used for judging debates |
| `ARGUING_MODEL` | `gpt-4o-mini` | Model used for generating AI arguments |
| `TEMPERATURE` | `0.7` | Creativity level for AI arguments |
| `JUDGE_TEMPERATURE` | `0.1` | Consistency level for judging |
| `MAX_NEW_TOKENS` | `300` | Max length for AI arguments |
| `MAX_ARGUMENT_LENGTH` | `300` | Max characters for user arguments |

### Supported Models

**OpenAI:**
- `gpt-4o-mini` (recommended)
- `gpt-4o`
- `gpt-3.5-turbo`

**Anthropic:**
- `claude-3-haiku-20240307`
- `claude-3-sonnet-20240229`

**OpenRouter:**
- `openrouter/anthropic/claude-3-haiku`
- `openrouter/openai/gpt-4o-mini`
- `openrouter/google/gemini-pro`
- `openrouter/meta-llama/llama-3-8b-instruct`

## 🏗️ Architecture

The demo is built with a modular architecture:

```
demo/
├── app.py              # Main Gradio interface
├── config.py           # Model configuration and API management
├── simple_models.py    # Simplified API-only model interfaces
├── debate_engine.py    # Core debate logic and topic management
├── leaderboard.py      # Scoring, ranking, and persistence
├── pyproject.toml      # uv project configuration
├── setup.sh           # Automated setup script
└── README.md          # This file
```

### Key Components

- **DebateEngine**: Handles topic selection, argument generation, and judging using prompts adapted from the main GRPO codebase
- **Leaderboard**: Implements round-robin tournament scoring with persistent storage
- **ModelManager**: Unified interface supporting multiple LLM providers with error handling
- **Gradio Interface**: Kid-friendly UI with real-time feedback and celebration animations

## 📊 Scoring System

The demo uses a sophisticated scoring system adapted from the main GRPO training codebase:

### Judge Evaluation Criteria
1. **Logical reasoning and evidence**
2. **Clear structure and organization**  
3. **Effective use of examples**
4. **Respectful tone**
5. **Addressing potential counterarguments**

### Format Scoring
- **XML Structure**: Points for proper `<reasoning>` and `<answer>` tags
- **Content Quality**: Penalties for extra content outside tags
- **Completeness**: Rewards for including both reasoning and final answer

### Leaderboard Ranking
- Primary: Win rate (wins ÷ total debates)
- Tiebreaker: Total number of debates
- Players need minimum 1 debate to appear on leaderboard
- AI baseline included for comparison

## 🎯 Educational Value

This demo teaches:
- **Structured Argumentation**: XML formatting encourages organized thinking
- **Critical Thinking**: Requirement to address counterarguments
- **Respectful Debate**: Judge evaluates tone and respect
- **Evidence-Based Reasoning**: Judge rewards logical reasoning and examples
- **Competition & Growth**: Leaderboard motivates improvement

## 🔧 Development

### Running Tests
```bash
python -m pytest
```

### Code Formatting
```bash
black .
flake8 .
```

### Adding New Topics
Edit the `TOPICS` list in `debate_engine.py` to add new debate topics.

### Customizing Prompts
The debate and judge prompts can be modified in `debate_engine.py` to change the evaluation criteria or argument structure.

## 🤝 Contributing

This demo is part of the larger GRPO debate training project. See the main repository for contribution guidelines.

## 📄 License

Same license as the main GRPO project.

## 🙏 Acknowledgments

- Built on the GRPO (Group Relative Policy Optimization) debate training framework
- Uses prompt engineering techniques from the main codebase's `evaluator.py` and `rldatasets.py`
- Designed for educational use at conferences and public demonstrations