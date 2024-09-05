import os
import tempfile
import uuid
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import TextSplitter
import requests

app = Flask(__name__)
CORS(app)

# Set up the API key and other configurations
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
api_url = "https://api.openai.com/v1/chat/completions"
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {OPENAI_API_KEY}',
}

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)

# Predefined HR Assistant parameters
HR_PDF_URL = "https://firebasestorage.googleapis.com/v0/b/voiage-67f40.appspot.com/o/pdfs%2Fdemo%2FHuman_Resources_Manual.pdf?alt=media"
HR_ROLE = "HR Assistant"
HR_CLASSES = {
    "HR-related": "Questions about HR regulations, policies, processes etc.",
    "Other Topic": "Inquiries/Statements non-related to HR",
    "Greeting": "Anything similar to Hey or How are you",
    "Not Understandable Word/Phrase": "Gibberish like eubcwucbi"
}
HR_AUTOMATIC_REPLIES = {
    "HR-related": "RAG",
    "Other Topic": "Unfortunately, I am unable to help you with that.",
    "Greeting": "Hello, I am an HR virtual assistant. How can I help you?",
    "Not Understandable Word/Phrase": "I apologize, I didn't quite get that. Could you ask again?"
}
HR_SEGMENT_ASSIGNMENTS = {
    "HR-related": "unified",
    "Other Topic": "unified",
    "Greeting": "unified",
    "Not Understandable Word/Phrase": "unified"
}

class ParagraphTextSplitter(TextSplitter):
    def split_text(self, text):
        return text.split('\n\n')

def download_pdf_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            return temp_file.name
    else:
        raise ValueError("Failed to download PDF from URL")

def prompt_func(query, n):
    class_list = ', '.join(f"'{c}'" for c in HR_CLASSES.keys())
    example_list = '. '.join([f"{cls}: {example}" for cls, example in HR_CLASSES.items()])
    
    if n == 1:
        prompt = (
            f"Role: {HR_ROLE}. Please classify the following query into one of the following categories: {class_list}. "
            f"Use the provided examples for accurate classification: {example_list}. "
            "Your response should ONLY be one of the categories provided, with no additional words."
            f"Query: {query}"
        )
    elif n == 2:
        prompt = "Rephrase the following sentence (make sure that your response has only the rephrased version and no additional words): I apologize, but I don't have the information you're looking for at the moment. Please let me know if there's anything else I can assist you with or if you have other HR-related questions."

    elif n == 3:
        prompt = "Rephrase the following sentence (make sure that your response has only the rephrased version and no additional words): Unfortunately, I am unable to help you with that. Please provide more specific questions related to HR topics."

    return prompt

def openaiAPI(prompt, temp, max_tokens=100):
    data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": "You are an AI user query classifier that is very experienced. You classify user inputs to one of the provided classes accurately"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temp,
    }
    response = requests.post(api_url, headers=headers, json=data)
    if response.status_code == 200:
        category = response.json()['choices'][0]['message']['content'].strip()
        return category
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def get_best_matching_text(query, index):
    prompt = prompt_func(query, 1)
    category = openaiAPI(prompt, 0.5)

    print(f"Classified category: {category}")

    if category in HR_CLASSES:
        if HR_AUTOMATIC_REPLIES.get(category) == "RAG":
            retriever = index.vectorstore.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
            result = qa_chain.run(query)
            if result in ("I don't know", "I don't know."):
                prompt = prompt_func(query, 2)
                result = openaiAPI(prompt, 0.9)
        else:
            reply = HR_AUTOMATIC_REPLIES.get(category, "I'm not sure how to respond to that.")
            prompt = "Rephrase the following text (your response should only have the rephrased text and no additional words): " + reply
            result = openaiAPI(prompt, 0.9)
    else:
        result = "Unfortunately, I am unable to help you with that. Please provide more specific questions related to HR topics."

    return result

# Global variable to store the indexed documents
global_index = None

def initialize_hr_assistant():
    global global_index
    if global_index is None:
        try:
            downloaded_pdf_path = download_pdf_from_url(HR_PDF_URL)
            loader = PyPDFLoader(downloaded_pdf_path)
            documents = loader.load()
            global_index = VectorstoreIndexCreator(
                vectorstore_cls=Chroma, 
                embedding=OpenAIEmbeddings(chunk_size=20),
                text_splitter=ParagraphTextSplitter()
            ).from_documents(documents)
            os.remove(downloaded_pdf_path)
            print("HR Assistant initialized successfully")
        except Exception as e:
            print(f"Error initializing HR Assistant: {str(e)}")
            global_index = None

@app.route('/initialize', methods=['GET'])
def initialize_chat():
    initialize_hr_assistant()
    return jsonify({"message": "HR Assistant initialized"}), 200

@app.route('/chat', methods=['POST'])
def chat():
    if global_index is None:
        return jsonify({"error": "HR Assistant not initialized"}), 400

    data = request.json
    query = data['query']

    try:
        response = get_best_matching_text(query, global_index)
        return jsonify({"response": response}), 200
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({"error": "An error occurred while processing your request"}), 500

if __name__ == '__main__':
    app.run(debug=True)