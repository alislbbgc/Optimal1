import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from streamlit_mic_recorder import speech_to_text
import fitz
import pdfplumber

# API Keys
groq_api_key = "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"
google_api_key = "AIzaSyDdAiOdIa2I28sphYw36Genb4D--2IN1tU"

# Page Configuration
st.set_page_config(
    page_title="BGC ChatBot",
    page_icon="BGC Logo Colored.svg",
    layout="wide"
)

# CSS Direction Handling
def apply_css_direction(direction):
    st.markdown(
        f"""
        <style>
            .stApp {{ direction: {direction}; text-align: {direction}; }}
            .stChatInput, .stChatMessage {{ direction: {direction}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Enhanced PDF Handler
class PDFManager:
    def __init__(self):
        self.current_pdf = None

    def get_pdf_metadata(self, pdf_path):
        try:
            with fitz.open(pdf_path) as doc:
                return {
                    'page_count': doc.page_count,
                    'is_valid': True
                }
        except Exception as e:
            return {'error': str(e)}

    def search_pdf(self, pdf_path, search_term):
        results = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and search_term in text:
                        results.append((page_num, text))
        except Exception as e:
            st.error(f"Search error: {str(e)}")
        return results

    def capture_pages(self, pdf_path, pages):
        outputs = []
        try:
            doc = fitz.open(pdf_path)
            meta = self.get_pdf_metadata(pdf_path)
            
            for page_info in pages:
                page_num, _ = page_info
                if 0 <= page_num < meta.get('page_count', 0):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap()
                    img_path = f"page_{page_num}.png"
                    pix.save(img_path)
                    outputs.append(img_path)
            doc.close()
        except Exception as e:
            st.error(f"Capture error: {str(e)}")
        return outputs

# Session State Initialization
if 'pdf_manager' not in st.session_state:
    st.session_state.pdf_manager = PDFManager()

# Sidebar Configuration
with st.sidebar:
    # Language Selection
    lang = st.selectbox("Interface Language", ["English", "العربية"])
    
    # Resource Mapping
    resources = {
        "English": {
            "pdf": "BGC-En.pdf",
            "embeddings": os.path.join("embeddings", "English", "embeddings"),
            "css": "ltr",
            "title": "Settings"
        },
        "العربية": {
            "pdf": "BGC-Ar.pdf",
            "embeddings": os.path.join("embeddings", "Arabic", "embeddings"),
            "css": "rtl",
            "title": "الإعدادات"
        }
    }
    
    # Apply Settings
    cfg = resources[lang]
    apply_css_direction(cfg['css'])
    st.title(cfg['title'])
    st.session_state.pdf_path = cfg['pdf']
    embeddings_path = cfg['embeddings']

    # API Initialization
    if groq_api_key and google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

        # Embeddings Loading with Language Check
        if "vectors" not in st.session_state or st.session_state.current_lang != lang:
            with st.spinner("Loading language resources..."):
                try:
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    st.session_state.vectors = FAISS.load_local(
                        embeddings_path,
                        embeddings,
                        allow_dangerous_deserialization=True
                    )
                    st.session_state.current_lang = lang
                except Exception as e:
                    st.error(f"Resource load error: {str(e)}")

        # Voice Input
        voice_input = speech_to_text(
            start_prompt="🎤",
            stop_prompt="⏹️ Stop" if lang == "English" else "⏹️ إيقاف",
            language="en" if lang == "English" else "ar",
            use_container_width=True
        )

        # Reset Functionality
        if st.button("Reset Chat" if lang == "English" else "إعادة تعيين"):
            st.session_state.messages = []
            st.session_state.memory.clear()
            st.rerun()

# Main Interface
col1, col2 = st.columns([1, 4])
with col1:
    st.image("BGC Logo Colored.svg", width=100)
with col2:
    if lang == "English":
        st.title("BGC ChatBot")
        st.write("""
        **Welcome!**  
        This chatbot uses information from the official BGC English documents.
        """)
    else:
        st.title("بوت الدردشة BGC")
        st.write("""
        **مرحبًا!**  
        يستخدم هذا البوت المعلومات من الوثائق العربية الرسمية لشركة غاز البصرة.
        """)

# Chat Memory Setup
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True
    )

# Full System Prompt
SYSTEM_PROMPT = """
You are a professional assistant for Basrah Gas Company (BGC). Follow these rules:

1. **Language Compliance**
   - Respond in the same language as the question
   - Use ONLY the {language} PDF document
   - Never reference other language resources

2. **Source Integrity**
   - Base answers strictly on {language} PDF content
   - Include page references from {language} PDF
   - If information isn't in the PDF, state: 
     - EN: "This information is not available in our English documents"
     - AR: "هذه المعلومات غير متوفرة في الوثائق العربية"

3. **Accuracy**
   - Be concise and factual
   - Never speculate or assume
   - Maintain professional tone

4. **Error Handling**
   - For unclear questions:
     - EN: "Could you please clarify your question?"
     - AR: "هل يمكنك توضيح سؤالك من فضلك؟"
   - For technical errors:
     - EN: "I'm experiencing technical difficulties"
     - AR: "أواجه صعوبات تقنية حاليا"
"""

# Response Generator
def generate_response(query):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT.format(language=lang)),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
        ("system", "Relevant Context: {context}")
    ])
    
    retriever = st.session_state.vectors.as_retriever()
    chain = create_retrieval_chain(
        retriever,
        create_stuff_documents_chain(llm, prompt_template)
    )
    
    return chain.invoke({
        "input": query,
        "context": retriever.get_relevant_documents(query),
        "history": st.session_state.memory.chat_memory.messages
    })

# Chat Handling
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input Processing
def process_input(input_text):
    st.session_state.messages.append({"role": "user", "content": input_text})
    
    with st.chat_message("user"):
        st.markdown(input_text)
    
    response = generate_response(input_text)
    
    if response:
        answer = response["answer"]
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        with st.chat_message("assistant"):
            st.markdown(answer)
        
        # Update Memory
        st.session_state.memory.save_context(
            {"input": input_text},
            {"output": answer}
        )
        
        # Show References
        if "context" in response:
            valid_pages = set()
            pdf_meta = st.session_state.pdf_manager.get_pdf_metadata(st.session_state.pdf_path)
            
            for doc in response["context"]:
                page = doc.metadata.get("page", "")
                if str(page).isdigit() and 0 <= int(page) < pdf_meta.get('page_count', 0):
                    valid_pages.add(int(page))
            
            if valid_pages:
                with st.expander("Page References" if lang == "English" else "مراجع الصفحات"):
                    pages = sorted(valid_pages)
                    st.write(f"Pages: {', '.join(map(str, pages))}" if lang == "English" else f"الصفحات: {', '.join(map(str, pages))}")
                    images = st.session_state.pdf_manager.capture_pages(st.session_state.pdf_path, [(p, "") for p in pages])
                    for img in images:
                        st.image(img)

# Handle Inputs
if voice_input:
    process_input(voice_input)

if prompt := st.chat_input("Type your question..." if lang == "English" else "اكتب سؤالك هنا..."):
    process_input(prompt)
