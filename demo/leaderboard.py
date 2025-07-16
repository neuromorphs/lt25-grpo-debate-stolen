"""
Leaderboard system for tracking debate performance.
Implements round-robin scoring and ranking logic.
"""

import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from debate_engine import DebateRound

@dataclass
class PlayerStats:
    """Statistics for a single player."""
    name: str
    total_debates: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_format_score: float = 0.0
    total_format_score: float = 0.0
    last_active: str = ""
    rounds_history: List[Dict] = None
    
    def __post_init__(self):
        if self.rounds_history is None:
            self.rounds_history = []
    
    def update_stats(self, debate_round: DebateRound):
        """Update stats with a new debate round."""
        self.total_debates += 1
        self.total_format_score += debate_round.user_format_score
        
        if debate_round.user_won:
            self.wins += 1
        else:
            self.losses += 1
            
        self.win_rate = self.wins / self.total_debates if self.total_debates > 0 else 0.0
        self.avg_format_score = self.total_format_score / self.total_debates if self.total_debates > 0 else 0.0
        self.last_active = datetime.now().isoformat()
        
        # Add to history (keep last 10 rounds)
        round_summary = {
            "topic": debate_round.topic,
            "stance": debate_round.stance,
            "won": debate_round.user_won,
            "format_score": debate_round.user_format_score,
            "timestamp": self.last_active
        }
        self.rounds_history.append(round_summary)
        if len(self.rounds_history) > 10:
            self.rounds_history = self.rounds_history[-10:]

class Leaderboard:
    """Manages the debate leaderboard and player statistics."""
    
    def __init__(self, data_file: str = "leaderboard_data.json"):
        self.data_file = data_file
        self.players: Dict[str, PlayerStats] = {}
        self.ai_baseline = PlayerStats(name="AI Baseline")
        self.load_data()
    
    def load_data(self):
        """Load leaderboard data from file."""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                # Load player stats
                for name, stats_dict in data.get("players", {}).items():
                    self.players[name] = PlayerStats(**stats_dict)
                
                # Load AI baseline
                if "ai_baseline" in data:
                    self.ai_baseline = PlayerStats(**data["ai_baseline"])
                    
            except Exception as e:
                print(f"Error loading leaderboard data: {e}")
                self.players = {}
                self.ai_baseline = PlayerStats(name="AI Baseline")
    
    def save_data(self):
        """Save leaderboard data to file."""
        try:
            data = {
                "players": {name: asdict(stats) for name, stats in self.players.items()},
                "ai_baseline": asdict(self.ai_baseline),
                "last_updated": datetime.now().isoformat()
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving leaderboard data: {e}")
    
    def add_debate_result(self, debate_round: DebateRound):
        """Add a debate result to the leaderboard."""
        player_name = debate_round.user_name
        
        # Initialize player if new
        if player_name not in self.players:
            self.players[player_name] = PlayerStats(name=player_name)
        
        # Update player stats
        self.players[player_name].update_stats(debate_round)
        
        # Update AI baseline (AI won if user lost)
        ai_round = DebateRound(
            topic=debate_round.topic,
            stance=debate_round.stance,
            user_name="AI",
            user_argument=debate_round.ai_argument,
            ai_argument="",
            judge_decision="ARGUMENT_1_WINS" if not debate_round.user_won else "ARGUMENT_2_WINS",
            judge_explanation="",
            user_won=not debate_round.user_won,
            user_format_score=debate_round.ai_format_score,
            ai_format_score=0.0
        )
        self.ai_baseline.update_stats(ai_round)
        
        # Save data
        self.save_data()
    
    def get_rankings(self, min_debates: int = 1) -> List[PlayerStats]:
        """Get player rankings sorted by win rate, then by total debates."""
        eligible_players = [
            player for player in self.players.values() 
            if player.total_debates >= min_debates
        ]
        
        # Sort by win rate (descending), then by total debates (descending)
        eligible_players.sort(
            key=lambda p: (p.win_rate, p.total_debates), 
            reverse=True
        )
        
        return eligible_players
    
    def get_top_players(self, limit: int = 10, min_debates: int = 1) -> List[PlayerStats]:
        """Get top players with minimum number of debates."""
        rankings = self.get_rankings(min_debates)
        return rankings[:limit]
    
    def get_player_stats(self, player_name: str) -> Optional[PlayerStats]:
        """Get stats for a specific player."""
        return self.players.get(player_name)
    
    def get_player_rank(self, player_name: str, min_debates: int = 1) -> Tuple[int, int]:
        """Get player's current rank and total eligible players."""
        rankings = self.get_rankings(min_debates)
        
        for i, player in enumerate(rankings):
            if player.name == player_name:
                return i + 1, len(rankings)
        
        return 0, len(rankings)
    
    def format_leaderboard_html(self, limit: int = 10) -> str:
        """Format leaderboard as HTML for Gradio display."""
        rankings = self.get_top_players(limit)
        
        if not rankings:
            return "<p>No debates completed yet. Be the first to join the leaderboard!</p>"
        
        html = """
        <div style="font-family: Arial, sans-serif;">
        <h3>🏆 Debate Leaderboard</h3>
        <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
        <thead>
        <tr style="background-color: #f0f0f0;">
        <th style="padding: 8px; border: 1px solid #ddd; text-align: center;">Rank</th>
        <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Player</th>
        <th style="padding: 8px; border: 1px solid #ddd; text-align: center;">Win Rate</th>
        <th style="padding: 8px; border: 1px solid #ddd; text-align: center;">Debates</th>
        <th style="padding: 8px; border: 1px solid #ddd; text-align: center;">Format Score</th>
        </tr>
        </thead>
        <tbody>
        """
        
        for i, player in enumerate(rankings):
            rank_icon = ""
            if i == 0:
                rank_icon = "🥇"
            elif i == 1:
                rank_icon = "🥈"  
            elif i == 2:
                rank_icon = "🥉"
            
            row_style = "background-color: #f9f9f9;" if i % 2 == 0 else ""
            
            html += f"""
            <tr style="{row_style}">
            <td style="padding: 8px; border: 1px solid #ddd; text-align: center;">{rank_icon} {i+1}</td>
            <td style="padding: 8px; border: 1px solid #ddd; text-align: left;"><strong>{player.name}</strong></td>
            <td style="padding: 8px; border: 1px solid #ddd; text-align: center;">{player.win_rate:.1%}</td>
            <td style="padding: 8px; border: 1px solid #ddd; text-align: center;">{player.wins}W-{player.losses}L</td>
            <td style="padding: 8px; border: 1px solid #ddd; text-align: center;">{player.avg_format_score:.2f}</td>
            </tr>
            """
        
        # Add AI baseline if we have data
        if self.ai_baseline.total_debates > 0:
            html += f"""
            <tr style="background-color: #ffe6e6; border-top: 2px solid #ff6666;">
            <td style="padding: 8px; border: 1px solid #ddd; text-align: center;">🤖</td>
            <td style="padding: 8px; border: 1px solid #ddd; text-align: left;"><strong>AI Baseline</strong></td>
            <td style="padding: 8px; border: 1px solid #ddd; text-align: center;">{self.ai_baseline.win_rate:.1%}</td>
            <td style="padding: 8px; border: 1px solid #ddd; text-align: center;">{self.ai_baseline.wins}W-{self.ai_baseline.losses}L</td>
            <td style="padding: 8px; border: 1px solid #ddd; text-align: center;">{self.ai_baseline.avg_format_score:.2f}</td>
            </tr>
            """
        
        html += """
        </tbody>
        </table>
        </div>
        """
        
        return html
    
    def get_player_summary(self, player_name: str) -> str:
        """Get a summary for a specific player."""
        player = self.get_player_stats(player_name)
        if not player:
            return f"Welcome, {player_name}! Start your first debate to join the leaderboard."
        
        rank, total = self.get_player_rank(player_name)
        
        summary = f"""
        <div style="font-family: Arial, sans-serif; background-color: #f0f8ff; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h4>📊 Your Stats, {player_name}</h4>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
        <div><strong>Current Rank:</strong> #{rank} of {total}</div>
        <div><strong>Win Rate:</strong> {player.win_rate:.1%}</div>
        <div><strong>Total Debates:</strong> {player.total_debates}</div>
        <div><strong>Record:</strong> {player.wins}W-{player.losses}L</div>
        </div>
        <div style="margin-top: 10px;"><strong>Avg Format Score:</strong> {player.avg_format_score:.2f}/1.0</div>
        </div>
        """
        
        return summary
    
    def clear_data(self):
        """Clear all leaderboard data."""
        self.players = {}
        self.ai_baseline = PlayerStats(name="AI Baseline")
        if os.path.exists(self.data_file):
            os.remove(self.data_file)

if __name__ == "__main__":
    # Test the leaderboard system
    from debate_engine import DebateRound
    
    leaderboard = Leaderboard("test_leaderboard.json")
    
    # Create test debate round
    test_round = DebateRound(
        topic="Schools should teach coding",
        stance="PRO", 
        user_name="TestUser",
        user_argument="<reasoning>Coding is important</reasoning><answer>We should teach coding</answer>",
        ai_argument="<reasoning>AI reasoning</reasoning><answer>AI agrees</answer>",
        judge_decision="ARGUMENT_1_WINS",
        judge_explanation="Human won because...",
        user_won=True,
        user_format_score=0.8,
        ai_format_score=0.9
    )
    
    leaderboard.add_debate_result(test_round)
    
    print("Leaderboard HTML:")
    print(leaderboard.format_leaderboard_html())
    
    print("\nPlayer Summary:")
    print(leaderboard.get_player_summary("TestUser"))
    
    # Clean up test file
    if os.path.exists("test_leaderboard.json"):
        os.remove("test_leaderboard.json")