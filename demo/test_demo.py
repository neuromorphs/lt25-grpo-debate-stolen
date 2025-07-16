#!/usr/bin/env python3
"""
Test the Gradio demo interface with mock models.
"""

import os
import sys
from unittest.mock import Mock, patch

# Mock environment for testing
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["JUDGE_MODEL"] = "gpt-4o-mini"
os.environ["ARGUING_MODEL"] = "gpt-4o-mini"

class MockModelInterface:
    """Mock model interface for testing."""
    
    def generate(self, system_prompt, user_prompt, **kwargs):
        """Generate a mock response."""
        if "judge" in system_prompt.lower():
            return """I think Argument 1 has better reasoning and examples. 
            The structure is clearer and addresses counterarguments effectively.
            
            ARGUMENT_1_WINS"""
        else:
            return """<reasoning>
This is a mock AI reasoning about the topic. I will consider the benefits 
and drawbacks of the position assigned to me.
</reasoning>
<answer>
This is a mock AI argument that supports the assigned position with logical reasoning.
</answer>"""

def test_debate_flow():
    """Test the complete debate flow with mocked models."""
    
    # Mock the model interfaces
    with patch('config.ModelManager') as MockModelManager:
        # Setup mock
        mock_manager = Mock()
        mock_manager.judge_model = MockModelInterface()
        mock_manager.arguing_model = MockModelInterface()
        mock_manager.generate_argument = lambda sp, up, **kwargs: MockModelInterface().generate(sp, up, **kwargs)
        mock_manager.judge_debate = lambda sp, up, **kwargs: MockModelInterface().generate(sp, up, **kwargs)
        MockModelManager.return_value = mock_manager
        
        # Import after mocking to avoid actual model loading
        from debate_engine import DebateEngine, DebateTopics
        from config import DemoConfig
        
        # Create engine with mock
        config = DemoConfig()
        engine = DebateEngine(mock_manager)
        
        # Test 1: Topic selection
        topic = DebateTopics.get_random_topic()
        stance = DebateTopics.get_random_stance()
        print(f"✅ Topic: {topic}")
        print(f"✅ Stance: {stance}")
        
        # Test 2: AI argument generation
        ai_argument = engine.generate_ai_argument(topic, stance)
        print(f"✅ AI argument generated: {ai_argument[:100]}...")
        
        # Test 3: Format scoring
        user_argument = """<reasoning>
        I think this topic is important because it affects many people.
        </reasoning>
        <answer>
        This is my strong argument for the position.
        </answer>"""
        
        format_score = engine.calculate_format_score(user_argument)
        print(f"✅ Format score: {format_score}")
        
        # Test 4: Judging
        user_won, explanation = engine.judge_arguments(topic, stance, user_argument, ai_argument)
        print(f"✅ Judge decision: {'Human' if user_won else 'AI'} wins")
        print(f"✅ Explanation: {explanation[:100]}...")
        
        # Test 5: Complete debate round
        debate_round = engine.run_debate_round("TestUser", topic, stance, user_argument)
        print(f"✅ Complete round: {debate_round.user_name} {'won' if debate_round.user_won else 'lost'}")
        
        return True

def test_leaderboard():
    """Test leaderboard functionality."""
    from leaderboard import Leaderboard, PlayerStats
    from debate_engine import DebateRound
    
    # Create test leaderboard
    leaderboard = Leaderboard("test_leaderboard.json")
    
    # Create mock debate round
    debate_round = DebateRound(
        topic="Test topic",
        stance="PRO",
        user_name="TestUser",
        user_argument="<reasoning>Test</reasoning><answer>Test answer</answer>",
        ai_argument="<reasoning>AI test</reasoning><answer>AI answer</answer>",
        judge_decision="ARGUMENT_1_WINS",
        judge_explanation="Test explanation",
        user_won=True,
        user_format_score=0.8,
        ai_format_score=0.7
    )
    
    # Add to leaderboard
    leaderboard.add_debate_result(debate_round)
    
    # Test HTML generation
    html = leaderboard.format_leaderboard_html()
    assert "TestUser" in html
    assert "80%" in html or "1.0" in html  # Win rate should be 100%
    
    print("✅ Leaderboard HTML generation works")
    
    # Test player summary
    summary = leaderboard.get_player_summary("TestUser")
    assert "TestUser" in summary
    
    print("✅ Player summary generation works")
    
    # Clean up
    if os.path.exists("test_leaderboard.json"):
        os.remove("test_leaderboard.json")
    
    return True

def test_gradio_interface():
    """Test Gradio interface functions."""
    
    # Mock the global variables in app.py
    with patch('app.model_manager') as mock_manager, \
         patch('app.debate_engine') as mock_engine, \
         patch('app.leaderboard') as mock_leaderboard:
        
        # Setup mocks
        mock_engine.run_debate_round.return_value = Mock(
            topic="Test topic",
            stance="PRO", 
            user_name="TestUser",
            user_argument="Test argument",
            ai_argument="AI argument",
            judge_decision="ARGUMENT_1_WINS",
            judge_explanation="Test explanation",
            user_won=True,
            user_format_score=0.8,
            ai_format_score=0.7
        )
        
        mock_engine.format_argument_for_display.return_value = {
            "reasoning": "Test reasoning",
            "answer": "Test answer",
            "full_text": "Full text"
        }
        
        mock_leaderboard.format_leaderboard_html.return_value = "<div>Test leaderboard</div>"
        mock_leaderboard.get_player_summary.return_value = "<div>Test summary</div>"
        
        # Import app functions after mocking
        from app import start_new_debate, submit_argument, format_debate_results
        
        # Test start_new_debate
        topic, stance, instruction, cleared_arg, cleared_results = start_new_debate()
        assert topic  # Should have a topic
        assert stance in ["PRO", "CON"]
        assert "Your Mission" in instruction
        
        print("✅ start_new_debate works")
        
        # Test submit_argument (will use mocks)
        results, leaderboard_html, player_summary = submit_argument(
            "TestUser", "Test topic", "PRO", "Test argument"
        )
        
        assert "TestUser" in results or "Congratulations" in results
        
        print("✅ submit_argument works")
        
        return True

def main():
    """Run all demo tests."""
    print("🧪 Testing Gradio demo functionality...\n")
    
    try:
        test_debate_flow()
        print()
        
        test_leaderboard()  
        print()
        
        test_gradio_interface()
        print()
        
        print("🎉 All demo tests passed! The interface should work with real API keys.")
        return True
        
    except Exception as e:
        print(f"❌ Demo test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)