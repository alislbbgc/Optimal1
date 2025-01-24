import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from streamlit_mic_recorder import speech_to_text
from PIL import Image
import time

# API Keys (Replace with your actual keys)
GROQ_API_KEY = "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"
GOOGLE_API_KEY = "AIzaSyDdAiOdIa2I28sphYw36Genb4D--2IN1tU"

# Load BGC Logo
bgc_logo = Image.open("BGC Logo.png")

# Styling Configuration
st.set_page_config(page_title="Mohammed Al-Yaseen | BGC ChatBot", page_icon=bgc_logo, layout="wide")

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer questions based on the provided context about Basrah Gas Company without explicitly mentioning the source of information."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
    ("system", "Context: {context}"),
])

# Voice Recording Function
def record_voice(language="en"):
    text = speech_to_text(
        start_prompt="🎤 Click and speak to ask a question",
        stop_prompt="⚠️ Stop recording 🚨",
        language=language,
        use_container_width=True,
        just_once=True,
    )
    return text if text else None

def init_llm():
    """Initialize LLM with error handling"""
    if not GROQ_API_KEY or not GOOGLE_API_KEY:
        st.error("Missing API keys. Please set GROQ_API_KEY and GOOGLE_API_KEY.")
        return None

    try:
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        return ChatGroq(groq_api_key=GROQ_API_KEY, model_name="gemma2-9b-it")
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return None

def display_chat_history():
    """Display chat history with custom styling"""
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        st.markdown(f'<div class="chat-message {role}">{content}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def process_user_input(user_input, llm):
    """Process user input and generate assistant response"""
    if "vectors" in st.session_state and st.session_state.vectors is not None:
        with st.spinner("Thinking..."):
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            response = retrieval_chain.invoke({
                "input": user_input,
                "context": retriever.get_relevant_documents(user_input),
                "history": st.session_state.memory.chat_memory.messages
            })

            assistant_response = response["answer"]

            st.session_state.memory.chat_memory.add_user_message(user_input)
            st.session_state.memory.chat_memory.add_ai_message(assistant_response)

            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_response}
            )
            st.markdown(f'<div class="chat-message assistant">{assistant_response}</div>', unsafe_allow_html=True)

            # Supporting Information
            with st.expander("Supporting Information"):
                if "context" in response:
                    for i, doc in enumerate(response["context"]):
                        page_number = doc.metadata.get("page", "unknown")
                        st.write(f"According to Page: {page_number}")
                        st.write(doc.page_content)
                        st.write("--------------------------------")
                else:
                    st.write("No context available.")
    else:
        assistant_response = "Error: Unable to load embeddings. Please check the embeddings folder."
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response}
        )
        st.markdown(f'<div class="chat-message assistant">{assistant_response}</div>', unsafe_allow_html=True)

def main():
    # Initialize LLM before using it
    llm = init_llm()
    if llm is None:
        st.stop()

    # Initialize Streamlit Sidebar
    with st.sidebar:
        st.title("Settings")
        voice_language = st.selectbox("Voice Input Language", ["English", "Arabic"])
        dark_mode = st.toggle("Dark Mode", value=True)

    # Initialize vectors
    if "vectors" not in st.session_state:
        with st.spinner("Loading embeddings... Please wait."):
            try:
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                embeddings_path = "embeddings"  # Path to your embeddings folder
                st.session_state.vectors = FAISS.load_local(
                    embeddings_path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                st.sidebar.write("Embeddings loaded successfully 🎉")
            except Exception as e:
                st.error(f"Error loading embeddings: {str(e)}")
                st.session_state.vectors = None

    # Initialize memory
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True
        )

    # Display BGC Logo
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    st.image(bgc_logo, width=200)
    st.markdown('</div>', unsafe_allow_html=True)

    st.title("Mohammed Al-Yaseen | BGC ChatBot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Clear chat history
    if st.button("Clear Chat History", key="clear-button"):
        st.session_state.messages = []
        st.session_state.memory.clear()

    # Display chat history
    display_chat_history()

    # Process user input
    input_lang_code = "ar" if voice_language == "Arabic" else voice_language.lower()[:2]

    # Sticky input at the bottom
    st.markdown('<div class="sticky-input">', unsafe_allow_html=True)

    # Create a container for the input and voice button
    st.markdown('<div class="input-container">', unsafe_allow_html=True)

    # Voice button (outside the form)
    voice_input = record_voice(language=input_lang_code)
    if voice_input:
        st.session_state.messages.append({"role": "user", "content": voice_input})
        st.markdown(f'<div class="chat-message user">{voice_input}</div>', unsafe_allow_html=True)
        process_user_input(voice_input, llm)

    # Form for text input
    with st.form(key="user_input_form", clear_on_submit=True):
        user_input = st.text_input("Ask something about the document", key="user_input", label_visibility="collapsed")
        submit_button = st.form_submit_button("Send")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Process text input
    if submit_button and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.markdown(f'<div class="chat-message user">{user_input}</div>', unsafe_allow_html=True)
        process_user_input(user_input, llm)

if __name__ == "__main__":
    main()
