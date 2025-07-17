
#!/usr/bin/env python3

import gradio as gr
import requests
import json

def chat_with_model(base_url, message, model_name=None):
    """
    Send a chat message to vLLM server and return the response.
    """
    url = f"{base_url}/v1/chat/completions"

    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "messages": [
            {
                "role": "user",
                "content": message
            }
        ],
        "temperature": 0,
        "max_tokens": 1024,
        "stream": False
    }

    if model_name:
        payload["model"] = model_name

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"
    except KeyError as e:
        return f"Error parsing response: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def create_chat_interface():
    """
    Create and configure the Gradio chat interface.
    """
    default_base_url = "http://isl-gpu39.rr.intel.com:8000"

    def chat_function(message, base_url=None, model_name=None):
        if not message.strip():
            return "Please enter a message."

        url = base_url if base_url else default_base_url
        response = chat_with_model(url, message, model_name)
        return response

    with gr.Blocks(title="vLLM Chat API Server") as interface:
        gr.Markdown("# vLLM Chat API Server")
        gr.Markdown("Connect to a vLLM server and chat with the model through this web interface.")

        with gr.Row():
            with gr.Column():
                base_url_input = gr.Textbox(
                    label="vLLM Server URL",
                    value=default_base_url,
                    placeholder="http://your-server:8000"
                )
                model_input = gr.Textbox(
                    label="Model Name (optional)",
                    placeholder="Leave empty for default model"
                )

        message_input = gr.Textbox(
            label="Your Message",
            placeholder="Type your message here...",
            lines=3
        )

        submit_btn = gr.Button("Send Message", variant="primary")

        response_output = gr.Textbox(
            label="Response",
            lines=10,
            interactive=False
        )

        submit_btn.click(
            fn=chat_function,
            inputs=[message_input, base_url_input, model_input],
            outputs=response_output
        )

        message_input.submit(
            fn=chat_function,
            inputs=[message_input, base_url_input, model_input],
            outputs=response_output
        )

    return interface

def main():
    """
    Launch the Gradio API server.
    """
    interface = create_chat_interface()

    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=False
    )

if __name__ == "__main__":
    main()

