import gradio as gr
import pandas as pd
import sys
import os

def load_question(question_id):
    """Load question text from questions.csv based on ID"""
    questions_file = os.path.join(os.path.dirname(__file__), 'questions.csv')
    try:
        df = pd.read_csv(questions_file)
        if 0 <= question_id < len(df):
            return df.iloc[question_id]['question'].strip()
        else:
            return f"Question {question_id} not found (valid range: 0-{len(df)-1})"
    except FileNotFoundError:
        return "Questions file not found"
    except Exception as e:
        return f"Error reading questions file: {str(e)}"

def load_leaderboard_data(question_id):
    """Load leaderboard data from responses CSV file"""
    filename = os.path.join(os.path.dirname(__file__), f"responses_{question_id}.csv")
    try:
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            if 'elo_rating' in df.columns and 'win_rate' in df.columns:
                # Sort by ELO rating descending
                df_sorted = df.sort_values('elo_rating', ascending=False).reset_index(drop=True)
                return df_sorted
            else:
                return pd.DataFrame(columns=['username', 'type', 'argument', 'win_rate', 'elo_rating'])
        else:
            return pd.DataFrame(columns=['username', 'type', 'argument', 'win_rate', 'elo_rating'])
    except Exception as e:
        print(f"Error loading leaderboard data: {e}")
        return pd.DataFrame(columns=['username', 'type', 'argument', 'win_rate', 'elo_rating'])

def create_leaderboard_html(df):
    """Create HTML table for leaderboard with icons"""
    if df.empty:
        return "<p>No leaderboard data available. Run evaluate_pairs.py first.</p>"
    
    html = """
    <div style="max-width: 1000px; margin: 0 auto;">
        <h2 style="text-align: center; color: #2d3748; margin-bottom: 20px;">🏆 Leaderboard</h2>
        <table style="width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <thead style="background: #4a5568; color: white;">
                <tr>
                    <th style="padding: 12px; text-align: left;">Rank</th>
                    <th style="padding: 12px; text-align: left;">Type</th>
                    <th style="padding: 12px; text-align: left;">Username</th>
                    <th style="padding: 12px; text-align: center;">ELO Rating</th>
                    <th style="padding: 12px; text-align: center;">Win Rate</th>
                    <th style="padding: 12px; text-align: left;">Argument</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for i, row in df.iterrows():
        rank = i + 1
        icon = "👤" if row['type'] == 'human' else "🤖"
        elo_rating = int(row['elo_rating']) if pd.notna(row['elo_rating']) else 1000
        win_rate = f"{row['win_rate']*100:.1f}%" if pd.notna(row['win_rate']) else "0.0%"
        argument_preview = str(row['argument'])[:150] + "..." if len(str(row['argument'])) > 150 else str(row['argument'])
        
        # Alternate row colors
        bg_color = "#f7fafc" if rank % 2 == 0 else "white"
        
        html += f"""
                <tr style="background: {bg_color}; border-bottom: 1px solid #e2e8f0;">
                    <td style="padding: 12px; font-weight: bold; color: #2d3748;">#{rank}</td>
                    <td style="padding: 12px; font-size: 20px;">{icon}</td>
                    <td style="padding: 12px; font-weight: 600; color: #2d3748;">{row['username']}</td>
                    <td style="padding: 12px; text-align: center; font-weight: bold; color: #4a5568;">{elo_rating}</td>
                    <td style="padding: 12px; text-align: center; font-weight: bold; color: #4a5568;">{win_rate}</td>
                    <td style="padding: 12px; color: #4a5568; font-style: italic;">{argument_preview}</td>
                </tr>
        """
    
    html += """
            </tbody>
        </table>
    </div>
    """
    
    return html

def create_leaderboard_dashboard(question_id):
    """Create Gradio leaderboard dashboard for the specified question"""
    question_text = load_question(question_id)
    
    def refresh_leaderboard():
        """Refresh the leaderboard data"""
        df = load_leaderboard_data(question_id)
        return create_leaderboard_html(df)
    
    with gr.Blocks(title=f"Question {question_id} Leaderboard") as demo:
        gr.Markdown(f"# Leaderboard: {question_text}")
        gr.Markdown("**Rankings based on ELO ratings from pairwise comparisons**")
        
        refresh_btn = gr.Button("🔄 Refresh Leaderboard", variant="primary")
        leaderboard_html = gr.HTML(value=refresh_leaderboard())
        
        refresh_btn.click(
            fn=refresh_leaderboard,
            outputs=[leaderboard_html]
        )
        
        # Load initial data
        demo.load(
            fn=refresh_leaderboard,
            outputs=[leaderboard_html]
        )
    
    return demo

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python leaderboard_dashboard.py <question_id>")
        sys.exit(1)
    
    try:
        question_id = int(sys.argv[1])
    except ValueError:
        print("Question ID must be an integer")
        sys.exit(1)
    
    demo = create_leaderboard_dashboard(question_id)
    demo.launch(share=True, server_port=7515)