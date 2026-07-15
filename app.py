import gradio as gr
from rag_engine import RAGChatbot

# Initialize our backend engine
bot = RAGChatbot()

def process_file(file):
    # When a file is uploaded, send it to the engine to be chunked and embedded
    status_message = bot.load_and_chunk_pdf(file.name)
    return status_message

def chat_interface(message, history):
    # When the user types a message, send it to the engine's query function
    return bot.answer_question(message, history)

# ---------- UI LAYOUT ----------
custom_theme = gr.themes.Base(
    primary_hue="slate",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_md,
)

# The theme parameter was removed from here
with gr.Blocks(title="RAG Chatbot") as app:
    gr.Markdown("""
    # 📚 RAG PDF Chatbot
    Upload a document, wait for the AI to ingest it into the vector database, and then ask questions about it. The bot will strictly answer using only the provided text.
    """)
    
    with gr.Row():
        # The file uploader
        file_upload = gr.File(label="1. Upload a PDF", file_types=[".pdf"])
        upload_status = gr.Textbox(label="System Status", value="Waiting for file...", interactive=False)
        
    # Link the upload action to our backend processing function
    file_upload.upload(fn=process_file, inputs=[file_upload], outputs=[upload_status])
    
    # Gradio's native Chat UI
    gr.Markdown("### 2. Chat with your Document")
    gr.ChatInterface(
        fn=chat_interface,
        examples=["What is this document about?", "Can you summarize the main points?"]
    )

if __name__ == "__main__":
    # The theme parameter was moved down here for Gradio 6.0+
    app.launch(theme=custom_theme)