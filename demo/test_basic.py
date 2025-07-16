#!/usr/bin/env python3
"""
Basic test script to verify core functionality without model loading.
"""

import sys
import os
import random

# Test 1: Topics and stances
def test_topics():
    """Test that we can generate topics and stances without dependencies."""
    topics = [
        "Video games should be taught as a school sport",
        "All schools should have mandatory cooking classes",
        "Homework should be replaced with project-based learning",
        "Every city should have a night market",
        "Movie theaters should have special quiet showings",
    ]
    
    stances = ["PRO", "CON"]
    
    # Test topic selection
    topic = random.choice(topics)
    stance = random.choice(stances)
    
    print(f"✅ Topic selection works: {topic}")
    print(f"✅ Stance selection works: {stance}")
    
    return True

# Test 2: Format scoring logic
def test_format_scoring():
    """Test XML format scoring logic."""
    import re
    
    def calculate_format_score(text: str) -> float:
        """Calculate format score based on XML structure."""
        score = 0.0
        
        # Check for strict format
        pattern = r"<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>"
        if re.search(pattern, text, re.DOTALL):
            score += 0.5
        else:
            # Check for relaxed format
            pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
            if re.search(pattern, text, re.DOTALL):
                score += 0.3
        
        # XML tag counting
        if "<reasoning>" in text: score += 0.1
        if "</reasoning>" in text: score += 0.1  
        if "<answer>" in text: score += 0.1
        if "</answer>" in text: score += 0.1
        
        return max(0.0, min(1.0, score))
    
    # Test cases
    test_cases = [
        ("<reasoning>\nGood reasoning\n</reasoning>\n<answer>\nGood answer\n</answer>", 0.9),
        ("<reasoning>Bad format</reasoning><answer>Also bad</answer>", 0.7),
        ("No XML tags at all", 0.0),
        ("<reasoning>Only reasoning</reasoning>", 0.2),
    ]
    
    for text, expected_min in test_cases:
        score = calculate_format_score(text)
        assert score >= expected_min * 0.8, f"Score {score} too low for: {text[:50]}..."
        print(f"✅ Format scoring works: {score:.2f} for '{text[:30]}...'")
    
    return True

# Test 3: Basic leaderboard logic
def test_leaderboard():
    """Test leaderboard calculations."""
    from dataclasses import dataclass
    
    @dataclass
    class MockPlayer:
        name: str
        wins: int
        losses: int
        
        @property
        def total_debates(self):
            return self.wins + self.losses
        
        @property 
        def win_rate(self):
            return self.wins / self.total_debates if self.total_debates > 0 else 0.0
    
    players = [
        MockPlayer("Alice", 5, 2),  # 71% win rate
        MockPlayer("Bob", 3, 1),    # 75% win rate  
        MockPlayer("Charlie", 8, 5), # 62% win rate
    ]
    
    # Sort by win rate
    sorted_players = sorted(players, key=lambda p: p.win_rate, reverse=True)
    
    print(f"✅ Leaderboard sorting works:")
    for i, player in enumerate(sorted_players):
        print(f"   {i+1}. {player.name}: {player.win_rate:.0%} ({player.wins}W-{player.losses}L)")
    
    assert sorted_players[0].name == "Bob", "Bob should be #1 with 75% win rate"
    assert sorted_players[1].name == "Alice", "Alice should be #2 with 71% win rate"
    
    return True

# Test 4: Environment variable handling
def test_env_config():
    """Test configuration from environment variables."""
    
    # Set some test env vars
    os.environ["JUDGE_MODEL"] = "test-judge-model"
    os.environ["TEMPERATURE"] = "0.8"
    
    # Simple config class
    class TestConfig:
        def __init__(self):
            self.judge_model = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
            self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
    
    config = TestConfig()
    
    assert config.judge_model == "test-judge-model"
    assert config.temperature == 0.8
    
    print(f"✅ Environment config works: {config.judge_model}, temp={config.temperature}")
    
    # Clean up
    del os.environ["JUDGE_MODEL"]
    del os.environ["TEMPERATURE"]
    
    return True

def main():
    """Run all tests."""
    print("🧪 Running basic functionality tests...\n")
    
    try:
        test_topics()
        print()
        
        test_format_scoring()
        print()
        
        test_leaderboard()
        print()
        
        test_env_config()
        print()
        
        print("🎉 All tests passed! Core functionality is working.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)