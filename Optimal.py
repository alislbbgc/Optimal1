import streamlit as st
import os
from langchain_groq import ChatGroq
from langdetect import detect, DetectorFactory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from streamlit_mic_recorder import speech_to_text
import fitz
import pdfplumber

# Initialize language detector first
DetectorFactory.seed = 0

# API Keys (replace with your actual keys)
groq_api_key = "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"
google_api_key = "AIzaSyDdAiOdIa2I28sphYw36Genb4D--2IN1tU"

# Configure page settings
st.set_page_config(
    page_title="BGC ChatBot",
    page_icon="BGC Logo Colored.svg",
    layout="wide"
)

def apply_css_direction(direction):
    st.markdown(
        f"""
        <style>
            .stApp {{ direction: {direction}; text-align: {direction}; }}
            .stChatInput {{ direction: {direction}; }}
            .stChatMessage {{ direction: {direction}; text-align: {direction}; }}
        </style>
        """,
        unsafe_allow_html=True,
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

# Initialize components
pdf_handler = PDFHandler()

# Sidebar configuration
with st.sidebar:
    st.title("Chat Controls")
    
    if groq_api_key and google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

        # Voice input component
        voice_input = speech_to_text(
            start_prompt="🎤",
            stop_prompt="⏹️ Stop",
            use_container_width=True,
            just_once=True,
            key="mic_button",
        )

        if st.button("Reset Conversation"):
            st.session_state.clear()
            st.rerun()

# Main interface
col1, col2 = st.columns([1, 4])
with col1:
    st.image("BGC Logo Colored.svg", width=100)
with col2:
    st.title("BGC Multilingual Assistant")
    st.write("""
    **Welcome!**  
    Ask questions in English or Arabic about company documents
    """)

# Initialize session state
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

# Core prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a technical documentation assistant for Basrah Gas Company. Strict rules:

    1. **Language Compliance:**
       - Respond EXCLUSIVELY in the user's input language
       - Never translate responses or mix languages
       - Maintain technical accuracy

    2. **Source Requirements:**
       - Use ONLY information from provided context
       - Never reference other sources
       - If no relevant context:
         - EN: "This information is not available in our documents"
         - AR: "هذه المعلومات غير متوفرة في الوثائق"

    3. **Formatting:**
       - Use markdown for technical clarity
       - Maintain proper text direction (RTL/LTR)
       - Use bullet points for lists
       - Bold key terms
    """),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
    ("system", "Document Context: {context}"),
])

def detect_input_language(text):
    try:
        return detect(text)
    except:
        return "en"

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
        st.error(f"Embeddings not found at: {embeddings_path}")
        return None
    except Exception as e:
        st.error(f"Loading Error: {str(e)}")
        return None

def process_query(user_input):
    lang = detect_input_language(user_input)
    apply_css_direction("rtl" if lang == "ar" else "ltr")
    
    # Load language-specific resources
    if "current_lang" not in st.session_state or st.session_state.current_lang != lang:
        st.session_state.vectors = load_embeddings(lang)
        st.session_state.current_lang = lang
    
    if not user_input.strip():
        return {
            "answer": "الرجاء إدخال سؤال صحيح" if lang == "ar" else "Please enter a valid question",
            "context": []
        }
    
    if not st.session_state.get("vectors"):
        return {
            "answer": "نظام التوثيق غير متوفر" if lang == "ar" else "Document system unavailable",
            "context": []
        }

    try:
        retriever = st.session_state.vectors.as_retriever()
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        context = retriever.get_relevant_documents(user_input)
        if not context:
            return {
                "answer": "لا توجد معلومات ذات صلة" if lang == "ar" else "No relevant information found",
                "context": []
            }

        return retrieval_chain.invoke({
            "input": user_input,
            "context": context,
            "history": st.session_state.memory.chat_memory.messages
        })
    except Exception as e:
        return {
            "answer": f"System Error: {str(e)}",
            "context": []
        }

# Handle voice input
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
            pdf_handler.current_pdf = pdf_handler.get_pdf_path(detect_input_language(voice_input))
            pages = {doc.metadata.get("page") for doc in response["context"] if doc.metadata.get("page") is not None}
            if pages:
                with st.expander("المراجع" if detect_input_language(voice_input) == "ar" else "References"):
                    st.write(f"الصفحات: {', '.join(map(str, sorted(pages)))}" if detect_input_language(voice_input) == "ar" 
                            else f"Pages: {', '.join(map(str, sorted(pages)))}")
                    for screenshot in pdf_handler.capture_screenshots(pages):
                        st.image(screenshot)

# Handle text input
user_input = st.chat_input("Type your question in English or Arabic...")
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
            lang = detect_input_language(user_input)
            pdf_handler.current_pdf = pdf_handler.get_pdf_path(lang)
            pages = {doc.metadata.get("page") for doc in response["context"] if doc.metadata.get("page") is not None}
            if pages:
                with st.expander("المراجع" if lang == "ar" else "References"):
                    st.write(f"الصفحات: {', '.join(map(str, sorted(pages)))}" if lang == "ar" 
                            else f"Pages: {', '.join(map(str, sorted(pages)))}")
                    for screenshot in pdf_handler.capture_screenshots(pages):
                        st.image(screenshot)
