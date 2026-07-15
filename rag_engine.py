import os
import PyPDF2
import chromadb
from dotenv import load_dotenv
from google import genai
from groq import Groq

# --- 1. LOAD BOTH API KEYS ---
load_dotenv()
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")

if not GOOGLE_KEY or not GROQ_KEY:
    raise ValueError("Error: Both GOOGLE_API_KEY and GROQ_API_KEY must be in your .env file.")

# --- 2. INITIALIZE BOTH CLOUD CLIENTS ---
google_client = genai.Client(api_key=GOOGLE_KEY)
groq_client = Groq(api_key=GROQ_KEY)

# Google does the math, Groq does the talking
embed_model = "gemini-embedding-2"
chat_model = "openai/gpt-oss-20b"

# Setup Vector Database 
chroma_client = chromadb.PersistentClient(path="./local_chroma_db")
collection = chroma_client.get_or_create_collection(name="pdf_knowledge_base")

class RAGChatbot:
    def __init__(self):
        self.document_loaded = False

    def load_and_chunk_pdf(self, file_path):
        if file_path is None:
            return "No file provided."
            
        print("Parsing PDF...")
        text = ""
        with open(file_path, 'rb') as f:
            pdf = PyPDF2.PdfReader(f)
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        
        chunk_size = 2000
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        
        print(f"Broke document into {len(chunks)} chunks. Generating embeddings via Google...")
        
        global collection
        chroma_client.delete_collection(name="pdf_knowledge_base")
        collection = chroma_client.get_or_create_collection(name="pdf_knowledge_base")
        
        for i, chunk in enumerate(chunks):
            print(f"Embedding chunk {i+1} of {len(chunks)}...")
            # Use Google to convert text to vectors
            embedding_response = google_client.models.embed_content(
                model=embed_model,
                contents=chunk
            )
            vector = embedding_response.embeddings[0].values
            
            collection.add(
                ids=[str(i)],
                embeddings=[vector],
                documents=[chunk]
            )
            
        self.document_loaded = True
        return "PDF successfully loaded! The Hybrid System is ready."

    def answer_question(self, user_question, history):
        if not self.document_loaded:
            return "Please upload a PDF first so I have something to read!"
            
        # 1. Use Google to embed the user's question
        question_response = google_client.models.embed_content(
            model=embed_model,
            contents=user_question
        )
        question_embedding = question_response.embeddings[0].values
        
        # 2. Search the database
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=3
        )
        
        retrieved_context = "\n\n".join(results['documents'][0])
        
        prompt = f"""
        You are a helpful assistant. Use ONLY the following Context to answer the user's Question. 
        If the answer is not contained in the Context, explicitly say "I cannot find the answer in the provided document." Do not guess.
        
        Context:
        {retrieved_context}
        
        Question:
        {user_question}
        """
        
        # 3. Use Groq to generate the answer instantly (Bypasses Google's 503 errors!)
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=chat_model,
        )
        
        return response.choices[0].message.content