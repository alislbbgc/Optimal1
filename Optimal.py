import streamlit as st
import os
from langchain_groq import ChatGroq
from langdetect import detect
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from streamlit_mic_recorder import speech_to_text
import fitz
import pdfplumber

# Force LTR direction for all elements
st.markdown("""
    <style>
    .stApp, .stChatInput, .stChatMessage {
        direction: ltr;
        text-align: left;
    }
    </style>
    """, unsafe_allow_html=True)

# API Configuration
groq_api_key = "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"
google_api_key = "AIzaSyDdAiOdIa2I28sphYw36Genb4D--2IN1tU"

st.set_page_config(
    page_title="BGC ChatBot",
    page_icon="BGC Logo Colored.svg",
    layout="wide"
)

class PDFHandler:
    def __init__(self):
        self.current_pdf = None
    
    def get_pdf_path(self, language):
        return "BGC-Ar.pdf" if language == "ar" else "BGC-En.pdf"
    
    def capture_screenshots(self, pages):
        if not self.current_pdf:
            return []
        
        doc = fitz.open(self.current_pdf)
        screenshots = []
        for page_number in pages:
            page = doc.load_page(page_number)
            pix = page.get_pixmap()
            screenshot_path = f"screenshot_page_{page_number}.png"
            pix.save(screenshot_path)
            screenshots.append(screenshot_path)
        return screenshots

pdf_handler = PDFHandler()

# Sidebar Configuration
with st.sidebar:
    st.title("Voice & Response Settings")
    
    # Language selection for voice input and responses
    response_language = st.selectbox(
        "Response Language",
        ["English", "Arabic"],
        index=0,
        key="lang_selector"
    )
    
    # Voice input component with selected language
    voice_input = speech_to_text(
        start_prompt="🎤 Start Recording",
        stop_prompt="⏹️ Stop Recording",
        language="ar" if response_language == "Arabic" else "en",
        use_container_width=True,
        just_once=True,
        key="mic_button",
    )

    if st.button("Clear Chat History"):
        st.session_state.clear()
        st.rerun()

# Main Interface
col1, col2 = st.columns([1, 4])
with col1:
    st.image("BGC Logo Colored.svg", width=100)
with col2:
    st.title("BGC Multilingual Assistant")
    st.write("""
    **Welcome!**  
    Ask questions using voice or text input
    """)

# Session State Management
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Core Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a technical documentation assistant for Basrah Gas Company. Rules:
    
    1. Response Language: {response_language}
    2. Use ONLY information from provided context
    3. If no relevant context, say "Information not found in documents"
    4. Maintain professional technical formatting
    """),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
    ("system", "Document Context: {context}"),
])

def load_embeddings(lang_code):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        lang_folder = "Arabic" if lang_code == "ar" else "English"
        embeddings_path = f"embeddings/{lang_folder}/embeddings"
        
        if os.path.exists(embeddings_path):
            return FAISS.load_local(
                embeddings_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
        return None
    except Exception as e:
        st.error(f"Loading Error: {str(e)}")
        return None

def process_query(user_input):
    lang_code = "ar" if st.session_state.lang_selector == "Arabic" else "en"
    
    # Load embeddings if not loaded or language changed
    if "vectors" not in st.session_state or st.session_state.current_lang != lang_code:
        st.session_state.vectors = load_embeddings(lang_code)
        st.session_state.current_lang = lang_code
    
    if not st.session_state.get("vectors"):
        return {
            "answer": "Document system unavailable",
            "context": []
        }

    try:
        retriever = st.session_state.vectors.as_retriever()
        document_chain = create_stuff_documents_chain(
            ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it"),
            prompt
        )
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        context = retriever.get_relevant_documents(user_input)
        return retrieval_chain.invoke({
            "input": user_input,
            "context": context,
            "history": st.session_state.memory.chat_memory.messages,
            "response_language": st.session_state.lang_selector
        })
    except Exception as e:
        return {
            "answer": f"System Error: {str(e)}",
            "context": []
        }

# Handle Voice Input
if voice_input:
    st.session_state.messages.append({"role": "user", "content": voice_input})
    with st.chat_message("user"):
        st.markdown(voice_input)
    
    response = process_query(voice_input)
    if response:
        assistant_response = response.get("answer", "")
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
        
        st.session_state.memory.chat_memory.add_user_message(voice_input)
        st.session_state.memory.chat_memory.add_ai_message(assistant_response)
        
        if response.get("context"):
            pdf_handler.current_pdf = pdf_handler.get_pdf_path("ar" if st.session_state.lang_selector == "Arabic" else "en")
            pages = {doc.metadata.get("page") for doc in response["context"] if doc.metadata.get("page") is not None}
            if pages:
                with st.expander("Document References"):
                    st.write(f"Pages: {', '.join(map(str, sorted(pages)))}")
                    for screenshot in pdf_handler.capture_screenshots(pages):
                        st.image(screenshot)

# Handle Text Input
user_input = st.chat_input("Type your question here...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    response = process_query(user_input)
    if response:
        assistant_response = response.get("answer", "")
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
        
        st.session_state.memory.chat_memory.add_user_message(user_input)
        st.session_state.memory.chat_memory.add_ai_message(assistant_response)
        
        if response.get("context"):
            pdf_handler.current_pdf = pdf_handler.get_pdf_path("ar" if st.session_state.lang_selector == "Arabic" else "en")
            pages = {doc.metadata.get("page") for doc in response["context"] if doc.metadata.get("page") is not None}
            if pages:
                with st.expander("Document References"):
                    st.write(f"Pages: {', '.join(map(str, sorted(pages)))}")
                    for screenshot in pdf_handler.capture_screenshots(pages):
                        st.image(screenshot)
