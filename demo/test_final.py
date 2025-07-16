#!/usr/bin/env python3
"""
Final test of the complete demo functionality.
"""

import os
import sys
from unittest.mock import patch, Mock

# Set mock environment for testing
os.environ["OPENAI_API_KEY"] = "test-key-for-demo"

def test_complete_demo():
    """Test the complete demo workflow."""
    
    # Mock OpenAI responses
    def mock_openai_create(**kwargs):
        mock_response = Mock()
        mock_choice = Mock()
        
        messages = kwargs.get('messages', [])
        if any('judge' in msg.get('content', '').lower() for msg in messages):
            # Judge response
            mock_choice.message.content = """The human argument shows better reasoning and structure. 
            It provides specific examples and addresses potential counterarguments effectively.
            
            ARGUMENT_1_WINS"""
        else:
            # AI argument response
            mock_choice.message.content = """<reasoning>
I need to argue for this position by considering the main benefits and addressing counterarguments.
The key points include practical advantages and positive social impact.
</reasoning>
<answer>
This position is beneficial because it would create positive outcomes for society and individuals through improved efficiency and enhanced quality of life.
</answer>"""
        
        mock_response.choices = [mock_choice]
        return mock_response
    
    with patch('openai.OpenAI') as MockOpenAI:
        mock_client = Mock()
        mock_client.chat.completions.create = mock_openai_create
        MockOpenAI.return_value = mock_client
        
        # Test imports
        from debate_engine import DebateEngine, DebateTopics, DebateRound
        from config import DemoConfig, ModelManager
        from leaderboard import Leaderboard
        from app import start_new_debate, submit_argument
        
        print("✅ All imports successful")
        
        # Test configuration
        config = DemoConfig()
        print(f"✅ Config loaded: {config.judge_model_name}")
        
        # Test model manager
        model_manager = ModelManager(config)
        print("✅ Model manager initialized")
        
        # Test debate engine
        engine = DebateEngine(model_manager)
        print("✅ Debate engine initialized")
        
        # Test topic selection
        topic = DebateTopics.get_random_topic()
        stance = DebateTopics.get_random_stance()
        print(f"✅ Random topic: {topic}")
        print(f"✅ Random stance: {stance}")
        
        # Test AI argument generation
        ai_argument = engine.generate_ai_argument(topic, stance)
        print(f"✅ AI argument generated: {ai_argument[:100]}...")
        
        # Test format scoring
        user_argument = """<reasoning>
This topic is important because it affects education and student development.
We should consider the benefits of practical skills and real-world application.
</reasoning>
<answer>
This position is beneficial because it provides students with essential life skills and practical knowledge that they can use immediately.
</answer>"""
        
        format_score = engine.calculate_format_score(user_argument)
        print(f"✅ Format score calculated: {format_score}")
        
        # Test judging
        user_won, explanation = engine.judge_arguments(topic, stance, user_argument, ai_argument)
        print(f"✅ Judge decision: {'Human' if user_won else 'AI'} wins")
        
        # Test complete debate round
        debate_round = engine.run_debate_round("TestUser", topic, stance, user_argument)
        print(f"✅ Complete debate round: {debate_round.user_name} {'won' if debate_round.user_won else 'lost'}")
        
        # Test leaderboard
        leaderboard = Leaderboard("test_final_leaderboard.json")
        leaderboard.add_debate_result(debate_round)
        
        html = leaderboard.format_leaderboard_html()
        assert "TestUser" in html
        print("✅ Leaderboard HTML generated successfully")
        
        summary = leaderboard.get_player_summary("TestUser")
        assert "TestUser" in summary
        print("✅ Player summary generated successfully")
        
        # Test Gradio interface functions
        topic, stance, instruction, cleared_arg, cleared_results = start_new_debate()
        assert topic and stance
        print("✅ start_new_debate function works")
        
        # Clean up
        if os.path.exists("test_final_leaderboard.json"):
            os.remove("test_final_leaderboard.json")
        
        return True

def test_app_launch():
    """Test that the app can be created without errors."""
    
    # Mock the global variables to avoid actual model loading
    with patch('app.model_manager'), \
         patch('app.debate_engine'), \
         patch('app.leaderboard'):
        
        from app import create_demo_interface
        
        # Create the interface (but don't launch)
        demo = create_demo_interface()
        
        print("✅ Gradio interface created successfully")
        print(f"✅ Demo title: {demo.title}")
        
        return True

def main():
    """Run final tests."""
    print("🏁 Final Demo Test\n")
    
    try:
        test_complete_demo()
        print()
        
        test_app_launch()
        print()
        
        print("🎉 ALL TESTS PASSED!")
        print("\nThe AI Debate Arena demo is ready!")
        print("\nNext steps:")
        print("1. Set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        print("2. Run the demo: uv run python app.py")
        print("3. Open http://localhost:7860 in your browser")
        
        return True
        
    except Exception as e:
        print(f"❌ Final test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)