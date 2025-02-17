import streamlit as st
import os
from typing import Dict, List, Optional, Union, Callable
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

# Initialize detector for consistent results
DetectorFactory.seed = 0

# Configuration
MAX_MESSAGES = 50
PDF_PATHS = {
    "ar": "BGC-Ar.pdf",
    "en": "BGC-En.pdf"
}

# Load API keys from environment variables
groq_api_key = os.getenv("GROQ_API_KEY", "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")
google_api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyDdAiOdIa2I28sphYw36Genb4D--2IN1tU")

# Streamlit Configuration
st.set_page_config(
    page_title="BGC ChatBot",
    page_icon="BGC Logo Colored.svg",
    layout="wide"
)

def apply_css_direction(direction: str) -> None:
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
        self.current_pdf: Optional[str] = None
        self.temp_dir = "temp_screenshots"
        
        # Create temp directory if it doesn't exist
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

    def get_pdf_path(self, language: str) -> str:
        return PDF_PATHS.get(language, PDF_PATHS["en"])

    def cleanup_old_screenshots(self):
        """Clean up old screenshot files"""
        if os.path.exists(self.temp_dir):
            for file in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    st.error(f"Error cleaning up file {file_path}: {e}")

    def capture_screenshots(self, pages: List[int]) -> List[str]:
        """Capture screenshots of PDF pages and return their paths"""
        screenshots = []
        if not self.current_pdf or not os.path.exists(self.current_pdf):
            st.error("PDF file not found")
            return screenshots

        # Clean up old screenshots
        self.cleanup_old_screenshots()

        try:
            doc = fitz.open(self.current_pdf)
            for page_number in pages:
                try:
                    screenshot_path = os.path.join(self.temp_dir, f"screenshot_page_{page_number}.png")
                    page = doc.load_page(page_number)
                    pix = page.get_pixmap()
                    pix.save(screenshot_path)
                    screenshots.append(screenshot_path)
                except Exception as e:
                    st.error(f"Error capturing screenshot for page {page_number}: {e}")
            return screenshots
        except Exception as e:
            st.error(f"Error opening PDF: {e}")
            return []
        finally:
            if 'doc' in locals():
                doc.close()

# Initialize components
pdf_handler = PDFHandler()

# Sidebar Configuration
with st.sidebar:
    st.title("Chat Controls")
    st.radio("Voice Input Language", ["English", "Arabic"], key="voice_lang")

    if groq_api_key and google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

        # Voice input component
        voice_input = speech_to_text(
            language='en' if st.session_state.voice_lang == "English" else 'ar',
            start_prompt="🎤",
            stop_prompt="⏹️ Stop",
            use_container_width=True,
            just_once=True,
            key="mic_button",
        )

        if st.button("Reset Conversation"):
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
    Ask questions in English or Arabic about company documents
    """)

# Session State Management
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# Manage message history size
if len(st.session_state.messages) > MAX_MESSAGES:
    st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]

# Chat History Display
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Core Prompt Template
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

def detect_input_language(text: str) -> str:
    try:
        return detect(text)
    except:
        return "en"

@st.cache_resource
def load_embeddings(lang_code: str) -> Optional[Union[FAISS, List[FAISS]]]:
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        lang_folder = "Arabic" if lang_code == "ar" else "English"

        if lang_code == "en":
            embeddings_paths = [
                f"embeddings/{lang_folder}/embeddings",
                f"embeddings/{lang_folder}/embeddingsOCR"
            ]
            all_vectors = []
            for embeddings_path in embeddings_paths:
                if os.path.exists(f"{embeddings_path}/index.pkl") and os.path.exists(f"{embeddings_path}/index.faiss"):
                    vectors = FAISS.load_local(
                        embeddings_path,
                        embeddings,
                        allow_dangerous_deserialization=True
                    )
                    all_vectors.append(vectors)
                else:
                    st.error(f"Embeddings not found at: {embeddings_path}")
                    return None
            return all_vectors
        else:
            embeddings_path = f"embeddings/{lang_folder}/embeddings"
            if os.path.exists(f"{embeddings_path}/index.pkl") and os.path.exists(f"{embeddings_path}/index.faiss"):
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

def process_query(user_input: str) -> Dict[str, Union[str, List]]:
    lang = detect_input_language(user_input)
    apply_css_direction("rtl" if lang == "ar" else "ltr")

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
        if lang == "en":
            retrievers = [v.as_retriever() for v in st.session_state.vectors]
            
            def combined_retrieval(query: str) -> List:
                all_docs = []
                for retriever in retrievers:
                    docs = retriever.get_relevant_documents(query)
                    all_docs.extend(docs)
                
                # Deduplication of documents
                unique_docs = []
                seen_docs = set()
                for doc in all_docs:
                    if doc.page_content not in seen_docs:
                        unique_docs.append(doc)
                        seen_docs.add(doc.page_content)
                return unique_docs

            retriever = combined_retrieval
        else:
            retriever = st.session_state.vectors.as_retriever()

        # Updated chain creation
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        # Get context directly instead of using retrieval chain
        context = retriever(user_input) if callable(retriever) else retriever.get_relevant_documents(user_input)
        
        if not context:
            return {
                "answer": "لا توجد معلومات ذات صلة" if lang == "ar" else "No relevant information found",
                "context": []
            }

        # Use document chain directly
        response = document_chain.invoke({
            "input": user_input,
            "context": context,
            "history": st.session_state.memory.chat_memory.messages
        })

        return {
            "answer": response,
            "context": context
        }
    except FileNotFoundError as e:
        return {
            "answer": "Document access error: " + str(e),
            "context": []
        }
    except Exception as e:
        return {
            "answer": "Unexpected error: " + str(e),
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
            lang = detect_input_language(voice_input)
            pdf_handler.current_pdf = pdf_handler.get_pdf_path(lang)
            pages = {doc.metadata.get("page") for doc in response["context"] if doc.metadata.get("page") is not None}
            if pages:
                with st.expander("المراجع" if lang == "ar" else "References"):
                    st.write(f"الصفحات: {', '.join(map(str, sorted(pages)))}" if lang == "ar"
                             else f"Pages: {', '.join(map(str, sorted(pages)))}")
                    screenshots = pdf_handler.capture_screenshots(pages)
                    for screenshot_path in screenshots:
                        try:
                            if os.path.exists(screenshot_path):
                                st.image(screenshot_path)
                            else:
                                st.error(f"Screenshot not found: {screenshot_path}")
                        except Exception as e:
                            st.error(f"Error displaying screenshot: {e}")

# Handle Text Input
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
                    screenshots = pdf_handler.capture_screenshots(pages)
                    for screenshot_path in screenshots:
                        try:
                            if os.path.exists(screenshot_path):
                                st.image(screenshot_path)
                            else:
                                st.error(f"Screenshot not found: {screenshot_path}")
                        except Exception as e:
                            st.error(f"Error displaying screenshot: {e}")
