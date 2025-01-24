import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from streamlit_mic_recorder import speech_to_text  # Import speech-to-text function

# Initialize API key variables
groq_api_key = "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"
google_api_key = "AIzaSyDdAiOdIa2I28sphYw36Genb4D--2IN1tU"

# Change the page title and icon
st.set_page_config(
    page_title="BGC ChatBot",  # Page title
    page_icon="🗨️",  # Page icon (you can change it to another icon)
    layout="wide"  # Page layout
)

# Sidebar configuration
with st.sidebar:
    st.title("Settings")

    # Validate API key inputs and initialize components if valid
    if groq_api_key and google_api_key:
        # Set Google API key as environment variable
        os.environ["GOOGLE_API_KEY"] = google_api_key

        # Initialize ChatGroq with the provided Groq API key
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

        # Define the chat prompt template with memory
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Attention Model: You are a specialized chatbot designed to assist individuals in the oil and gas industry, with a particular focus on content related to the Basrah Gas Company (BGC). "
    "Your responses must primarily rely on the PDF files uploaded by the user, which contain information specific to the oil and gas sector and BGC's operational procedures. "
    "If a specific answer cannot be directly found in the PDFs, you are permitted to provide a logical and well-reasoned response based on your internal knowledge. "
    "Under no circumstances should you use or rely on information from external sources, including the internet.\n\n"
    "Guidelines:\n"
    "1. **Primary Source Referencing:**\n"
    "- Always base your responses on the information from the uploaded PDFs. "
    "If the PDFs contain partial or related information, integrate it with logical reasoning to provide a comprehensive response. "
    "Do not explicitly mention the source (e.g., page numbers) unless the user specifically asks for it.\n\n"
    "2. **Logical Reasoning:**\n"
    "- When specific answers are unavailable in the PDFs, use your internal knowledge to provide logical, industry-relevant responses. "
    "Do not state that the response is based on reasoning unless explicitly asked.\n\n"
    "3. **Visual Representation:**\n"
    "- When users request visual representations (e.g., diagrams, charts, or illustrations), create accurate and relevant visuals based on the uploaded PDF content and logical reasoning. "
    "Ensure the visuals align precisely with the context provided and are helpful for understanding the topic.\n\n"
    "4. **Restricted Data Usage:**\n"
    "- Avoid using or assuming information from external sources, including the internet or any pre-existing external knowledge that falls outside the uploaded materials or your internal logical reasoning.\n\n"
    "5. **Professional and Contextual Responses:**\n"
    "- Ensure responses remain professional, accurate, and relevant to the oil and gas industry, with particular tailoring for Basrah Gas Company. "
    "Maintain a helpful, respectful, and clear tone throughout your interactions.\n\n"
    "6. **Multilingual Support:**\n"
    "- Detect the language of the user's input (Arabic or English) and respond in the same language. "
    "If the input is in Arabic, provide the response in Arabic. If the input is in English, provide the response in English.\n\n"
    "Expected Output:\n"
    "- Precise and accurate answers derived from the uploaded PDFs, without explicitly mentioning the source unless asked.\n"
    "- Logical and well-reasoned responses when direct answers are not available in the PDFs, without explicitly stating that the response is based on reasoning.\n"
    "- Accurate visual representations (when requested) based on PDF content or logical reasoning.\n"
    "- Polite acknowledgments when information is unavailable in the provided material, coupled with logical insights where possible.\n"
    "- Responses in the same language as the user's input (Arabic or English).\n\n"
    "Thank you for your accuracy, professionalism, and commitment to providing exceptional assistance tailored to the Basrah Gas Company and the oil and gas industry."),
            MessagesPlaceholder(variable_name="history"),  # Add chat history to the prompt
            ("human", "{input}"),
            ("system", "Context: {context}"),
        ])

        # Load existing embeddings from files
        if "vectors" not in st.session_state:
            with st.spinner("Loading embeddings... Please wait."):
                # Initialize embeddings
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001"
                )

                # Load existing FAISS index with safe deserialization
                embeddings_path = "embeddings"  # Path to your embeddings folder
                try:
                    st.session_state.vectors = FAISS.load_local(
                        embeddings_path,
                        embeddings,
                        allow_dangerous_deserialization=True  # Only use if you trust the source of the embeddings
                    )
                    st.sidebar.write("Embeddings loaded successfully 🎉")
                except Exception as e:
                    st.error(f"Error loading embeddings: {str(e)}")
                    st.session_state.vectors = None

        # Voice language selection
        voice_language = st.selectbox("Voice Input Language", ["English", "Arabic"])

        # Microphone button in the sidebar
        st.markdown("### Voice Input")
        input_lang_code = "ar" if voice_language == "Arabic" else "en"  # Set language code
        voice_input = speech_to_text(
            start_prompt="🎤",
            stop_prompt="⏹️ Stop",
            language=input_lang_code,  # Language (en for English, ar for Arabic)
            use_container_width=True,
            just_once=True,
            key="mic_button",
        )
    else:
        st.error("Please enter both API keys to proceed.")

# Main area for chat interface
st.title("Mohammed Al-Yaseen | BGC ChatBot")

# Initialize session state for chat messages if not already done
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize memory if not already done
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True
    )

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# If voice input is detected, process it
if voice_input:
    st.session_state.messages.append({"role": "user", "content": voice_input})
    with st.chat_message("user"):
        st.markdown(voice_input)

    if "vectors" in st.session_state and st.session_state.vectors is not None:
        # Create and configure the document chain and retriever
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Get response from the assistant
        response = retrieval_chain.invoke({
            "input": voice_input,
            "context": retriever.get_relevant_documents(voice_input),
            "history": st.session_state.memory.chat_memory.messages  # Include chat history
        })
        assistant_response = response["answer"]

        # Append and display assistant's response
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response}
        )
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

        # Add user and assistant messages to memory
        st.session_state.memory.chat_memory.add_user_message(voice_input)
        st.session_state.memory.chat_memory.add_ai_message(assistant_response)

        # Display supporting information (page numbers only)
        with st.expander("Supporting Information"):
            if "context" in response:
                # Extract unique page numbers from the context
                page_numbers = set()
                for doc in response["context"]:
                    page_number = doc.metadata.get("page", "unknown")
                    if page_number != "unknown" and str(page_number).isdigit():  # Check if page_number is a valid number
                        page_numbers.add(int(page_number))  # Convert to integer for sorting

                # Display the page numbers
                if page_numbers:
                    page_numbers_str = ", ".join(map(str, sorted(page_numbers)))  # Sort pages numerically and convert back to strings
                    st.write(f"This answer is according to pages: {page_numbers_str}")
                else:
                    st.write("No valid page numbers available in the context.")
            else:
                st.write("No context available.")
    else:
        # Prompt user to ensure embeddings are loaded
        assistant_response = (
            "Embeddings not loaded. Please check if the embeddings path is correct."
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response}
        )
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

# Text input field
human_input = st.chat_input("Type your question here...")

# If text input is detected, process it
if human_input:
    st.session_state.messages.append({"role": "user", "content": human_input})
    with st.chat_message("user"):
        st.markdown(human_input)

    if "vectors" in st.session_state and st.session_state.vectors is not None:
        # Create and configure the document chain and retriever
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Get response from the assistant
        response = retrieval_chain.invoke({
            "input": human_input,
            "context": retriever.get_relevant_documents(human_input),
            "history": st.session_state.memory.chat_memory.messages  # Include chat history
        })
        assistant_response = response["answer"]

        # Append and display assistant's response
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response}
        )
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

        # Add user and assistant messages to memory
        st.session_state.memory.chat_memory.add_user_message(human_input)
        st.session_state.memory.chat_memory.add_ai_message(assistant_response)

        # Display supporting information (page numbers only)
        with st.expander("Page References"):
            if "context" in response:
                # Extract unique page numbers from the context
                page_numbers = set()
                for doc in response["context"]:
                    page_number = doc.metadata.get("page", "unknown")
                    if page_number != "unknown" and str(page_number).isdigit():  # Check if page_number is a valid number
                        page_numbers.add(int(page_number))  # Convert to integer for sorting

                # Display the page numbers
                if page_numbers:
                    page_numbers_str = ", ".join(map(str, sorted(page_numbers)))  # Sort pages numerically and convert back to strings
                    st.write(f"This answer is according to pages: {page_numbers_str}")
                else:
                    st.write("No valid page numbers available in the context.")
            else:
                st.write("No context available.")
    else:
        # Prompt user to ensure embeddings are loaded
        assistant_response = (
            "Embeddings not loaded. Please check if the embeddings path is correct."
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response}
        )
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
