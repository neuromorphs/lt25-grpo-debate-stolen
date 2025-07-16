"""
Main Gradio interface for the AI Debate Arena.
A fun, interactive debate game for kids and general public.
"""

import gradio as gr
import random
from typing import Tuple, Dict, Any
import os

from config import DemoConfig, ModelManager, load_config_from_env
from debate_engine import DebateEngine, DebateTopics, DebateRound
from leaderboard import Leaderboard

# Global variables
config = load_config_from_env()
model_manager = None
debate_engine = None
leaderboard = Leaderboard()

def initialize_models():
    """Initialize models with error handling."""
    global model_manager, debate_engine
    
    try:
        model_manager = ModelManager(config)
        debate_engine = DebateEngine(model_manager)
        return "✅ Models loaded successfully!"
    except Exception as e:
        error_msg = f"❌ Error loading models: {str(e)}"
        
        # Check for common API key issues
        if "OPENAI_API_KEY" in str(e):
            error_msg += "\n\n💡 Please set your OPENAI_API_KEY environment variable."
        elif "ANTHROPIC_API_KEY" in str(e):
            error_msg += "\n\n💡 Please set your ANTHROPIC_API_KEY environment variable."
        elif "OPENROUTER_API_KEY" in str(e):
            error_msg += "\n\n💡 Please set your OPENROUTER_API_KEY environment variable."
        
        return error_msg

def start_new_debate() -> Tuple[str, str, str, str, str]:
    """Start a new debate round with random topic and stance."""
    topic = DebateTopics.get_random_topic()
    stance = DebateTopics.get_random_stance()
    
    instruction = f"""
    🎯 **Your Mission:** Argue for the **{stance}** position!
    
    📝 **Format your argument like this:**
    ```
    <reasoning>
    Your step-by-step thinking process here...
    </reasoning>
    <answer>
    Your final 2-3 sentence argument here
    </answer>
    ```
    
    💡 **Tips:**
    - Use examples and evidence
    - Address potential counterarguments  
    - Keep it respectful and clear
    - Stay under {config.max_argument_length} characters
    """
    
    return (
        topic,  # Current topic
        stance,  # Current stance  
        instruction,  # Instructions
        "",  # Clear argument input
        ""   # Clear results
    )

def submit_argument(
    user_name: str, 
    topic: str, 
    stance: str, 
    user_argument: str
) -> Tuple[str, str, str]:
    """Submit user argument and run the debate."""
    
    if not model_manager or not debate_engine:
        return "❌ Models not loaded. Please initialize first.", "", ""
    
    if not user_name.strip():
        return "❌ Please enter your name first!", "", ""
    
    if not user_argument.strip():
        return "❌ Please write your argument!", "", ""
    
    if len(user_argument) > config.max_argument_length:
        return f"❌ Argument too long! Please keep it under {config.max_argument_length} characters.", "", ""
    
    try:
        # Run the debate
        debate_round = debate_engine.run_debate_round(
            user_name=user_name.strip(),
            topic=topic,
            stance=stance,
            user_argument=user_argument
        )
        
        # Add to leaderboard
        leaderboard.add_debate_result(debate_round)
        
        # Format results
        results = format_debate_results(debate_round)
        leaderboard_html = leaderboard.format_leaderboard_html()
        player_summary = leaderboard.get_player_summary(user_name.strip())
        
        return results, leaderboard_html, player_summary
        
    except Exception as e:
        error_msg = f"❌ Error during debate: {str(e)}"
        return error_msg, "", ""

def format_debate_results(debate_round: DebateRound) -> str:
    """Format the debate results for display."""
    
    # Format user argument
    user_formatted = debate_engine.format_argument_for_display(debate_round.user_argument)
    ai_formatted = debate_engine.format_argument_for_display(debate_round.ai_argument)
    
    # Determine winner styling
    if debate_round.user_won:
        user_style = "background-color: #d4edda; border-left: 5px solid #28a745;"
        ai_style = "background-color: #f8d7da; border-left: 5px solid #dc3545;"
        result_emoji = "🎉"
        result_text = "Congratulations! You won this debate!"
    else:
        user_style = "background-color: #f8d7da; border-left: 5px solid #dc3545;"
        ai_style = "background-color: #d4edda; border-left: 5px solid #28a745;"
        result_emoji = "🤖"
        result_text = "The AI won this round. Try again!"
    
    results_html = f"""
    <div style="font-family: Arial, sans-serif; margin: 20px 0;">
        
        <div style="text-align: center; margin: 20px 0; padding: 15px; background-color: #f0f8ff; border-radius: 10px;">
            <h2>{result_emoji} {result_text}</h2>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
            
            <div style="padding: 15px; border-radius: 8px; {user_style}">
                <h3>👤 Your Argument ({debate_round.stance})</h3>
                <div style="margin: 10px 0;">
                    <strong>Reasoning:</strong>
                    <p style="font-style: italic; margin: 5px 0;">{user_formatted['reasoning']}</p>
                </div>
                <div style="margin: 10px 0;">
                    <strong>Final Answer:</strong>
                    <p style="font-weight: bold; margin: 5px 0;">{user_formatted['answer']}</p>
                </div>
                <div style="margin: 10px 0; font-size: 0.9em; color: #666;">
                    Format Score: {debate_round.user_format_score:.2f}/1.0
                </div>
            </div>
            
            <div style="padding: 15px; border-radius: 8px; {ai_style}">
                <h3>🤖 AI Argument ({debate_round.stance})</h3>
                <div style="margin: 10px 0;">
                    <strong>Reasoning:</strong>
                    <p style="font-style: italic; margin: 5px 0;">{ai_formatted['reasoning']}</p>
                </div>
                <div style="margin: 10px 0;">
                    <strong>Final Answer:</strong>
                    <p style="font-weight: bold; margin: 5px 0;">{ai_formatted['answer']}</p>
                </div>
                <div style="margin: 10px 0; font-size: 0.9em; color: #666;">
                    Format Score: {debate_round.ai_format_score:.2f}/1.0
                </div>
            </div>
            
        </div>
        
        <div style="padding: 15px; background-color: #fff3cd; border-radius: 8px; margin: 20px 0;">
            <h3>⚖️ Judge's Decision</h3>
            <div style="background-color: white; padding: 10px; border-radius: 5px; margin: 10px 0;">
                {debate_round.judge_explanation.replace(chr(10), '<br>')}
            </div>
        </div>
        
    </div>
    """
    
    return results_html

def get_format_feedback(argument: str) -> str:
    """Get real-time feedback on argument formatting."""
    if not debate_engine:
        return "Models not loaded yet."
    
    return debate_engine.get_format_feedback(argument)

def create_demo_interface():
    """Create the main Gradio interface."""
    
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title=config.title,
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .debate-header {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
        }
        """
    ) as demo:
        
        # Header
        gr.HTML(f"""
        <div class="debate-header">
            <h1>{config.title}</h1>
            <p style="font-size: 1.2em; margin: 10px 0;">{config.description}</p>
            <p>💡 Format your arguments with &lt;reasoning&gt; and &lt;answer&gt; tags for the best scores!</p>
        </div>
        """)
        
        # Model status
        with gr.Row():
            model_status = gr.HTML(value="🔄 Initializing models...")
            
        # Player info and controls
        with gr.Row():
            with gr.Column(scale=2):
                user_name = gr.Textbox(
                    label="👤 Your Name", 
                    placeholder="Enter your name to join the leaderboard",
                    value=""
                )
            with gr.Column(scale=1):
                new_debate_btn = gr.Button("🎲 New Debate Topic", variant="primary", size="lg")
        
        # Current debate info
        with gr.Row():
            with gr.Column():
                current_topic = gr.Textbox(
                    label="📋 Current Topic", 
                    interactive=False,
                    value="Click 'New Debate Topic' to start!"
                )
                current_stance = gr.Textbox(
                    label="🎯 Your Position", 
                    interactive=False,
                    value=""
                )
                
        # Instructions
        instructions = gr.HTML(value="")
        
        # Argument input
        with gr.Row():
            user_argument = gr.Textbox(
                label=f"✍️ Your Argument (max {config.max_argument_length} chars)",
                placeholder="Write your argument here using <reasoning> and <answer> tags...",
                lines=8,
                max_lines=10
            )
        
        # Real-time feedback and submit
        with gr.Row():
            with gr.Column(scale=3):
                format_feedback = gr.HTML(value="")
            with gr.Column(scale=1):
                submit_btn = gr.Button("🚀 Submit Argument", variant="primary", size="lg")
        
        # Results area
        debate_results = gr.HTML(value="")
        
        # Player stats and leaderboard
        with gr.Row():
            with gr.Column(scale=1):
                player_summary = gr.HTML(value="")
            with gr.Column(scale=2):
                leaderboard_display = gr.HTML(value=leaderboard.format_leaderboard_html())
        
        # Initialize models when the demo loads
        demo.load(initialize_models, outputs=[model_status])
        
        # Event handlers
        new_debate_btn.click(
            start_new_debate,
            outputs=[current_topic, current_stance, instructions, user_argument, debate_results]
        )
        
        submit_btn.click(
            submit_argument,
            inputs=[user_name, current_topic, current_stance, user_argument],
            outputs=[debate_results, leaderboard_display, player_summary]
        )
        
        # Real-time format feedback
        user_argument.change(
            get_format_feedback,
            inputs=[user_argument],
            outputs=[format_feedback]
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
            <p style="color: #666; margin: 0;">
                🤖 Powered by AI • Built with Gradio • 
                <a href="https://github.com/neuromorphs/lt25-grpo-debate-stolen" target="_blank">View Source</a>
            </p>
        </div>
        """)
    
    return demo

if __name__ == "__main__":
    # Create and launch the demo
    demo = create_demo_interface()
    
    # Launch configuration
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Standard Gradio port
        share=False,            # Set to True to create public link
        show_error=True,
        debug=True
    )