import pandas as pd
import itertools
import argparse
import numpy as np
import requests
import json

def load_template():
    with open('judge_prompt.txt', 'r') as f:
        return f.read().strip()

def evaluate_pair(base_url, template, response1, response2, topic="General", model_name=None):
    prompt = template.format(
        debate_mode="PRO",
        topic=topic,
        arg1_response=response1,
        arg2_response=response2
    )
    
    url = f"{base_url}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0,
        "max_tokens": 1024,
        "stream": False
    }
    
    if model_name:
        payload["model"] = model_name
    
    try:
        print(prompt)
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        print(f"Request error evaluating pair: {e}")
        return None
    except KeyError as e:
        print(f"Error parsing response: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error evaluating pair: {e}")
        return None

def calculate_elo_ratings(win_matrix, usernames, initial_rating=1000, k_factor=32):
    ratings = {username: initial_rating for username in usernames}
    
    for i, username1 in enumerate(usernames):
        for j, username2 in enumerate(usernames):
            if i != j and win_matrix[i][j] + win_matrix[j][i] > 0:
                total_games = win_matrix[i][j] + win_matrix[j][i]
                wins = win_matrix[i][j]
                
                for _ in range(total_games):
                    expected1 = 1 / (1 + 10**((ratings[username2] - ratings[username1]) / 400))
                    actual = 1 if wins > 0 else 0
                    ratings[username1] += k_factor * (actual - expected1)
                    
                    expected2 = 1 / (1 + 10**((ratings[username1] - ratings[username2]) / 400))
                    actual2 = 1 if wins == 0 else 0
                    ratings[username2] += k_factor * (actual2 - expected2)
                    
                    if wins > 0:
                        wins -= 1
    
    return ratings

def calculate_win_rates_and_elo(df, topic, base_url, model_name=None):
    template = load_template()
    
    # Get all unique responses
    responses = df['argument'].tolist()
    usernames = df['username'].tolist()
    unique_usernames = list(set(usernames))
    n_users = len(unique_usernames)
    
    # Initialize win matrix and counts
    win_matrix = np.zeros((n_users, n_users), dtype=int)
    username_to_idx = {username: i for i, username in enumerate(unique_usernames)}
    
    win_counts = {username: 0 for username in unique_usernames}
    total_comparisons = {username: 0 for username in unique_usernames}
    
    # Evaluate all pairs
    for i, j in itertools.combinations(range(len(responses)), 2):
        response1 = responses[i]
        response2 = responses[j]
        username1 = usernames[i]
        username2 = usernames[j]
        
        idx1 = username_to_idx[username1]
        idx2 = username_to_idx[username2]
        
        result = evaluate_pair(base_url, template, response1, response2, topic, model_name)
        print(result)
        
        if result == "ARGUMENT_1_WINS":
            win_counts[username1] += 1
            win_matrix[idx1][idx2] += 1
        elif result == "ARGUMENT_2_WINS":
            win_counts[username2] += 1
            win_matrix[idx2][idx1] += 1
        
        total_comparisons[username1] += 1
        total_comparisons[username2] += 1
        
        print(f"Evaluated {username1} vs {username2}: {result}")
    
    # Calculate win rates
    win_rates = {}
    for username in unique_usernames:
        if total_comparisons[username] > 0:
            win_rates[username] = win_counts[username] / total_comparisons[username]
        else:
            win_rates[username] = 0.0
    
    # Calculate ELO ratings
    elo_ratings = calculate_elo_ratings(win_matrix, unique_usernames)
    
    return win_rates, elo_ratings, win_matrix, unique_usernames

def load_question_topic(question_id):
    questions_df = pd.read_csv('questions.csv')
    if question_id < len(questions_df):
        return questions_df.iloc[question_id]['question']
    else:
        return "General"

def display_leaderboard(elo_ratings, win_rates, win_matrix, usernames):
    print("\n" + "="*60)
    print("🏆 LEADERBOARD 🏆")
    print("="*60)
    
    # Sort by ELO rating
    sorted_users = sorted(usernames, key=lambda x: elo_ratings[x], reverse=True)
    
    print(f"{'Rank':<4} {'Username':<20} {'ELO Rating':<12} {'Win Rate':<10} {'Games':<6}")
    print("-" * 60)
    
    for i, username in enumerate(sorted_users, 1):
        elo = int(elo_ratings[username])
        win_rate = win_rates[username] * 100
        
        # Count total games for this user
        user_idx = usernames.index(username)
        total_games = np.sum(win_matrix[user_idx, :]) + np.sum(win_matrix[:, user_idx])
        
        print(f"{i:<4} {username:<20} {elo:<12} {win_rate:<10.1f}% {total_games:<6}")
    
    print("="*60)
    
    # Display win matrix
    print("\n📊 HEAD-TO-HEAD WIN MATRIX")
    print("="*40)
    print("Rows = winner, Columns = opponent")
    print(f"{'Username':<15}", end="")
    for username in usernames:
        print(f"{username[:8]:<9}", end="")
    print()
    
    for i, username in enumerate(usernames):
        print(f"{username[:14]:<15}", end="")
        for j in range(len(usernames)):
            if i == j:
                print(f"{'--':<9}", end="")
            else:
                print(f"{win_matrix[i][j]:<9}", end="")
        print()

def main(question_id, base_url, model_name=None):
    # Load question topic
    topic = load_question_topic(question_id)
    
    # Load CSV
    csv_filename = f'responses_{question_id}.csv'
    df = pd.read_csv(csv_filename)
    
    # Calculate win rates and ELO ratings
    win_rates, elo_ratings, win_matrix, unique_usernames = calculate_win_rates_and_elo(df, topic, base_url, model_name)
    
    # Add win_rate and elo_rating columns
    df['win_rate'] = df['username'].map(win_rates)
    df['elo_rating'] = df['username'].map(elo_ratings)
    
    # Save updated CSV
    df.to_csv(csv_filename, index=False)
    print(f"Win rates and ELO ratings calculated and saved to {csv_filename}")
    print(f"Topic: {topic}")
    
    # Display leaderboard
    display_leaderboard(elo_ratings, win_rates, win_matrix, unique_usernames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate debate responses')
    parser.add_argument('question_id', type=int, help='Question ID to evaluate')
    parser.add_argument('--vllm-url', required=True, help='vLLM server URL (e.g., http://localhost:8000)')
    parser.add_argument('--model', help='Model name (optional)')
    args = parser.parse_args()
    main(args.question_id, args.vllm_url, args.model)
