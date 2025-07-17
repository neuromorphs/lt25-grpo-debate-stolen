import gradio as gr
import pandas as pd
import sys
import os
import csv

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

def save_response(username, message, question_id):
    """Save response to responses_[question_id].csv"""
    filename = os.path.join(os.path.dirname(__file__), f"responses_{question_id}.csv")
    
    try:
        # Check if file exists to determine if we need to write header
        file_exists = os.path.exists(filename)
        
        # Clean newline characters from the message
        cleaned_message = message.replace('\n', ' ').replace('\r', ' ')
        
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['username', 'type', 'argument'])
            writer.writerow([username, 'human', cleaned_message])
        
        return f"Response saved to {os.path.basename(filename)}"
    except (IOError, csv.Error) as e:
        return f"Error saving response: {str(e)}"

def create_submission_dashboard(question_id):
    """Create Gradio submission dashboard for the specified question"""
    question_text = load_question(question_id)
    
    def submit_response(username, message):
        if not username.strip() or not message.strip():
            gr.Info("Please fill in both username and message fields.")
            return
        save_response(username, message, question_id)
        gr.Info("Response submitted successfully!")
    
    def update_char_count(text):
        return f"Characters: {len(text)}"
    
    with gr.Blocks(title=f"Question {question_id} Submission") as demo:
        gr.Markdown(f"# {question_text}")
        gr.Markdown("**Please write an argument in support of the question above.**")
        
        username_input = gr.Textbox(label="Username", placeholder="Enter your username")
        message_input = gr.Textbox(label="Message", placeholder="Enter your response", lines=5)
        char_count_label = gr.Markdown("Characters: 0")
        submit_btn = gr.Button("Submit Response", variant="primary")
        
        message_input.change(
            fn=update_char_count,
            inputs=[message_input],
            outputs=[char_count_label]
        )
        
        submit_btn.click(
            fn=submit_response,
            inputs=[username_input, message_input],
            outputs=None
        )
    
    return demo

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python submission_dashboard.py <question_id>")
        sys.exit(1)
    
    try:
        question_id = int(sys.argv[1])
    except ValueError:
        print("Question ID must be an integer")
        sys.exit(1)
    
    demo = create_submission_dashboard(question_id)
    demo.launch(share=True, server_port=7514)