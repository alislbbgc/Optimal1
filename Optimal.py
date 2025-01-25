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

# Initialize API key variables
groq_api_key = "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"
google_api_key = "AIzaSyDdAiOdIa2I28sphYw36Genb4D--2IN1tU"

# Change the page title and icon
st.set_page_config(
    page_title="BGC ChatBot",
    page_icon="BGC Logo Colored.svg",
    layout="wide"
)

def apply_css_direction(direction):
    st.markdown(
        f"""
        <style>
            .stApp {{
                direction: {direction};
                text-align: {direction};
            }}
            .stChatInput {{
                direction: {direction};
            }}
            .stChatMessage {{
                direction: {direction};
                text-align: {direction};
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

class PDFSearchAndDisplay:
    def __init__(self):
        pass

    def search_and_highlight(self, pdf_path, search_term):
        highlighted_pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages):
                text = page.extract_text()
                if search_term in text:
                    highlighted_pages.append((page_number, text))
        return highlighted_pages

    def capture_screenshots(self, pdf_path, pages):
        doc = fitz.open(pdf_path)
        screenshots = []
        for page_number, _ in pages:
            page = doc.load_page(page_number)
            pix = page.get_pixmap()
            screenshot_path = f"screenshot_page_{page_number}.png"
            pix.save(screenshot_path)
            screenshots.append(screenshot_path)
        return screenshots

# Sidebar configuration
with st.sidebar:
    interface_language = st.selectbox("Interface Language", ["English", "العربية"])
    
    if interface_language == "العربية":
        apply_css_direction("rtl")
        st.title("الإعدادات")
    else:
        apply_css_direction("ltr")
        st.title("Settings")

    if groq_api_key and google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

        # Handle language change
        if "current_language" in st.session_state:
            if st.session_state.current_language != interface_language:
                if "vectors" in st.session_state:
                    del st.session_state.vectors
                if "memory" in st.session_state:
                    del st.session_state.memory
                st.session_state.current_language = interface_language
        else:
            st.session_state.current_language = interface_language

        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are a helpful assistant for Basrah Gas Company (BGC). Follow these rules strictly:

            1. **Language Handling:**
               - Respond in the language of the user's question
               - If asked for specific language, use that language

            2. **Source Restriction:**
               - Use ONLY the provided {language} documents for answers
               - Never use information from other languages

            3. **Contextual Answers:**
               - Base answers strictly on provided context
               - If unsure, say:
                 - EN: "I don't have enough information from English documents"
                 - AR: "لا توجد معلومات كافية من الوثائق العربية"

            4. **Professional Tone:**
               - Maintain formal, technical language
               - Use industry-specific terminology
            """),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
            ("system", "Context: {context}"),
        ])

        if "vectors" not in st.session_state:
            loading_text = "جارٍ تحميل التضميدات..." if interface_language == "العربية" else "Loading embeddings..."
            with st.spinner(loading_text):
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                language_folder = "Arabic" if st.session_state.current_language == "العربية" else "English"
                embeddings_path = f"embeddings/{language_folder}/embeddings"
                
                try:
                    st.session_state.vectors = FAISS.load_local(
                        embeddings_path,
                        embeddings,
                        allow_dangerous_deserialization=True
                    )
                except Exception as e:
                    error_msg = f"خطأ في التحميل: {str(e)}" if interface_language == "العربية" else f"Loading error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.vectors = None

        # Voice input components
        input_lang_code = "ar" if interface_language == "العربية" else "en"
        voice_input = speech_to_text(
            start_prompt="🎤",
            stop_prompt="⏹️ إيقاف" if interface_language == "العربية" else "⏹️ Stop",
            language=input_lang_code,
            use_container_width=True,
            just_once=True,
            key="mic_button",
        )

        if st.button("إعادة تعيين" if interface_language == "العربية" else "Reset"):
            st.session_state.messages = []
            if "memory" in st.session_state:
                st.session_state.memory.clear()
            st.rerun()

# Determine PDF path based on current language
language_folder = "Arabic" if st.session_state.get("current_language", "English") == "العربية" else "English"
pdf_path = "BGC-Ar.pdf" if language_folder == "Arabic" else "BGC-En.pdf"
pdf_searcher = PDFSearchAndDisplay()

# Main chat interface
col1, col2 = st.columns([1, 4])
with col1:
    st.image("BGC Logo Colored.svg", width=100)
with col2:
    if interface_language == "العربية":
        st.title("محمد الياسين | بوت الدردشة BGC")
        st.write("""
        **مرحبًا!**  
        هذا بوت الدردشة الخاص بغاز البصرة. الأسئلة تجيب فقط من الوثائق العربية.
        """)
    else:
        st.title("Mohammed Al-Yaseen | BGC ChatBot")
        st.write("""
        **Welcome!**  
        This BGC chatbot answers exclusively from English documents.
        """)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True
    )

negative_phrases = [
    "I'm sorry", "عذرًا", "لا أملك معلومات", "I don't have information",
    "لم أتمكن من فهم", "I couldn't understand", "الوثائق العربية", "English documents"
]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def process_input(user_input):
    if "vectors" in st.session_state and st.session_state.vectors:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        response = retrieval_chain.invoke({
            "input": user_input,
            "context": retriever.get_relevant_documents(user_input),
            "history": st.session_state.memory.chat_memory.messages
        })
        
        return response
    return None

# Handle voice input
if voice_input:
    st.session_state.messages.append({"role": "user", "content": voice_input})
    with st.chat_message("user"):
        st.markdown(voice_input)
    
    response = process_input(voice_input)
    if response:
        assistant_response = response["answer"]
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
        
        st.session_state.memory.chat_memory.add_user_message(voice_input)
        st.session_state.memory.chat_memory.add_ai_message(assistant_response)
        
        if not any(phrase in assistant_response for phrase in negative_phrases):
            with st.expander("مراجع" if interface_language == "العربية" else "References"):
                page_numbers = set()
                for doc in response["context"]:
                    page = doc.metadata.get("page", "")
                    if str(page).isdigit():
                        page_numbers.add(int(page))
                
                if page_numbers:
                    pages_str = ", ".join(map(str, sorted(page_numbers)))
                    source_note = f"المصدر: الصفحات {pages_str}" if interface_language == "العربية" else f"Source: Pages {pages_str}"
                    st.write(source_note)
                    
                    screenshots = pdf_searcher.capture_screenshots(pdf_path, [(p, "") for p in page_numbers])
                    for ss in screenshots:
                        st.image(ss)

# Handle text input
input_placeholder = "اكتب سؤالك..." if interface_language == "العربية" else "Type your question..."
human_input = st.chat_input(input_placeholder)
if human_input:
    st.session_state.messages.append({"role": "user", "content": human_input})
    with st.chat_message("user"):
        st.markdown(human_input)
    
    response = process_input(human_input)
    if response:
        assistant_response = response["answer"]
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
        
        st.session_state.memory.chat_memory.add_user_message(human_input)
        st.session_state.memory.chat_memory.add_ai_message(assistant_response)
        
        if not any(phrase in assistant_response for phrase in negative_phrases):
            with st.expander("مراجع" if interface_language == "العربية" else "References"):
                page_numbers = set()
                for doc in response["context"]:
                    page = doc.metadata.get("page", "")
                    if str(page).isdigit():
                        page_numbers.add(int(page))
                
                if page_numbers:
                    pages_str = ", ".join(map(str, sorted(page_numbers)))
                    source_note = f"المصدر: الصفحات {pages_str}" if interface_language == "العربية" else f"Source: Pages {pages_str}"
                    st.write(source_note)
                    
                    screenshots = pdf_searcher.capture_screenshots(pdf_path, [(p, "") for p in page_numbers])
                    for ss in screenshots:
                        st.image(ss)
