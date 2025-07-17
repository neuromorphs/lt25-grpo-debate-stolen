import argparse
import pandas as pd
from gradio_client import Client

def load_question_topic(question_id):
    questions_df = pd.read_csv('questions.csv')
    if question_id < len(questions_df):
        return questions_df.iloc[question_id]['question']
    else:
        return "General"

def main():
    parser = argparse.ArgumentParser(description='Generate inference for debate question')
    parser.add_argument('question_id', type=int, help='Question ID to generate response for')
    parser.add_argument('--message', type=str, default=None, help='Custom message (optional)')
    args = parser.parse_args()
    
    # Load question topic
    topic = load_question_topic(args.question_id)
    
    # Use custom message or default to the topic
    message = args.message if args.message else f"Provide an argument for: {topic}"
    
    client = Client("https://7064048a4490f5e9cb.gradio.live")
    result = client.predict(
        message=message,
        api_name="/chat_function"
    )
    
    print(f"Question ID: {args.question_id}")
    print(f"Topic: {topic}")
    print(f"Response: {result}")

if __name__ == "__main__":
    main()
