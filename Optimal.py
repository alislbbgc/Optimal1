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

# Initialize API keys
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
        return "BGC-Ar.pdf" if language == "العربية" else "BGC-En.pdf"
    
    def capture_screenshots(self, pages):
        if not self.current_pdf:
            return []
        
        doc = fitz.open(self.current_pdf)
        screenshots = []
        for page_number, _ in pages:
            page = doc.load_page(page_number)
            pix = page.get_pixmap()
            screenshot_path = f"screenshot_page_{page_number}.png"
            pix.save(screenshot_path)
            screenshots.append(screenshot_path)
        return screenshots

# Initialize PDF handler
pdf_handler = PDFHandler()

# Sidebar configuration
with st.sidebar:
    # Language selection
    interface_language = st.selectbox("Interface Language", ["English", "العربية"])
    
    # Apply CSS direction
    text_direction = "rtl" if interface_language == "العربية" else "ltr"
    apply_css_direction(text_direction)
    
    # Set sidebar title
    sidebar_title = "الإعدادات" if interface_language == "العربية" else "Settings"
    st.title(sidebar_title)

    # Initialize API keys
    if groq_api_key and google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

        # Handle language changes
        if "current_language" not in st.session_state:
            st.session_state.current_language = interface_language
        elif st.session_state.current_language != interface_language:
            st.session_state.current_language = interface_language
            # Reset vectors and memory on language change
            keys_to_reset = ["vectors", "memory", "messages"]
            for key in keys_to_reset:
                if key in st.session_state:
                    del st.session_state[key]

        # Load embeddings
        if "vectors" not in st.session_state:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            language_folder = "Arabic" if interface_language == "العربية" else "English"
            embeddings_path = f"embeddings/{language_folder}/embeddings"
            
            if os.path.exists(embeddings_path):
                try:
                    st.session_state.vectors = FAISS.load_local(
                        embeddings_path,
                        embeddings,
                        allow_dangerous_deserialization=True
                    )
                except Exception as e:
                    error_msg = f"خطأ في تحميل البيانات: {str(e)}" if interface_language == "العربية" else f"Loading error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.vectors = None
            else:
                st.error("مسار التضميدات غير موجود" if interface_language == "العربية" else "Embeddings path not found")

        # Voice input component
        voice_input = speech_to_text(
            start_prompt="🎤",
            stop_prompt="⏹️ إيقاف" if interface_language == "العربية" else "⏹️ Stop",
            language="ar" if interface_language == "العربية" else "en",
            use_container_width=True,
            just_once=True,
            key="mic_button",
        )

        # Reset button
        if st.button("إعادة تعيين" if interface_language == "العربية" else "Reset Chat"):
            st.session_state.messages = []
            if "memory" in st.session_state:
                st.session_state.memory.clear()
            st.rerun()

# Main chat interface
col1, col2 = st.columns([1, 4])
with col1:
    st.image("BGC Logo Colored.svg", width=100)
with col2:
    if interface_language == "العربية":
        st.title("بوت الدردشة BGC")
        st.write("""
        **مرحبًا!**  
        هذا البوت يستخدم وثائق الشركة العربية فقط للإجابة على الأسئلة.
        """)
    else:
        st.title("BGC ChatBot")
        st.write("""
        **Welcome!**  
        This chatbot answers using only English company documents.
        """)

# Initialize chat memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True
    )

# Initialize messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Define the validated prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a professional assistant for Basrah Gas Company. Follow these rules:

    1. **Language Handling:**
       - Respond in the user's question language
       - Maintain technical accuracy

    2. **Source Requirements:**
       - Use ONLY the provided context
       - Never invent information

    3. **Uncertain Responses:**
       - If context is insufficient, respond:
         - EN: "This information is not available in our English documents"
         - AR: "هذه المعلومات غير متوفرة في الوثائق العربية"

    4. **Formatting:**
       - Use bullet points for lists
       - Bold important terms
       - Maintain professional tone
    """),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
    ("system", "Relevant Context: {context}"),
])

# Process user input
def handle_user_input(user_input):
    if not user_input.strip():
        return {
            "answer": "الرجاء إدخال سؤال صحيح" if interface_language == "العربية" 
                     else "Please enter a valid question",
            "context": []
        }
    
    if "vectors" not in st.session_state or not st.session_state.vectors:
        return {
            "answer": "نظام البحث غير جاهز" if interface_language == "العربية" 
                     else "Search system not ready",
            "context": []
        }

    try:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        context_docs = retriever.get_relevant_documents(user_input)
        if not context_docs:
            return {
                "answer": "لا توجد معلومات ذات صلة" if interface_language == "العربية" 
                         else "No relevant information found",
                "context": []
            }

        return retrieval_chain.invoke({
            "input": user_input,
            "context": context_docs,
            "history": st.session_state.memory.chat_memory.messages
        })
    except Exception as e:
        return {
            "answer": f"خطأ في النظام: {str(e)}" if interface_language == "العربية" 
                     else f"System error: {str(e)}",
            "context": []
        }

# Handle voice input
if voice_input:
    st.session_state.messages.append({"role": "user", "content": voice_input})
    with st.chat_message("user"):
        st.markdown(voice_input)
    
    response = handle_user_input(voice_input)
    if response:
        assistant_response = response.get("answer", "")
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
        
        # Update memory
        st.session_state.memory.chat_memory.add_user_message(voice_input)
        st.session_state.memory.chat_memory.add_ai_message(assistant_response)
        
        # Handle document references
        if response.get("context"):
            pdf_handler.current_pdf = pdf_handler.get_pdf_path(interface_language)
            page_numbers = {doc.metadata.get("page") for doc in response["context"] if doc.metadata.get("page") is not None}
            
            if page_numbers:
                with st.expander("المراجع" if interface_language == "العربية" else "References"):
                    pages_str = ", ".join(map(str, sorted(page_numbers)))
                    st.write(f"الصفحات: {pages_str}" if interface_language == "العربية" else f"Pages: {pages_str}")
                    
                    screenshots = pdf_handler.capture_screenshots([(p, "") for p in page_numbers])
                    for ss in screenshots:
                        st.image(ss)

# Handle text input
input_placeholder = "اكتب سؤالك هنا..." if interface_language == "العربية" else "Type your question here..."
user_input = st.chat_input(input_placeholder)
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    response = handle_user_input(user_input)
    if response:
        assistant_response = response.get("answer", "")
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
        
        # Update memory
        st.session_state.memory.chat_memory.add_user_message(user_input)
        st.session_state.memory.chat_memory.add_ai_message(assistant_response)
        
        # Handle document references
        if response.get("context"):
            pdf_handler.current_pdf = pdf_handler.get_pdf_path(interface_language)
            page_numbers = {doc.metadata.get("page") for doc in response["context"] if doc.metadata.get("page") is not None}
            
            if page_numbers:
                with st.expander("المراجع" if interface_language == "العربية" else "References"):
                    pages_str = ", ".join(map(str, sorted(page_numbers)))
                    st.write(f"الصفحات: {pages_str}" if interface_language == "العربية" else f"Pages: {pages_str}")
                    
                    screenshots = pdf_handler.capture_screenshots([(p, "") for p in page_numbers])
                    for ss in screenshots:
                        st.image(ss)
