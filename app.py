import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import tempfile

# Initialize API key variables
groq_api_key = "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"
google_api_key = "AIzaSyDdAiOdIa2I28sphYw36Genb4D--2IN1tU"

# Sidebar configuration
with st.sidebar:
    # Validate API key inputs and initialize components if valid
    if groq_api_key and google_api_key:
        # Set Google API key as environment variable
        os.environ["GOOGLE_API_KEY"] = google_api_key

        # Initialize ChatGroq with the provided Groq API key
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

        # Define the chat prompt template
        prompt = ChatPromptTemplate.from_template(
            """
            Answer the questions based on the provided context only.
            Please provide the most accurate response based on the question.
            <context>
            {context}
            <context>
            Questions: {input}
            """
        )

        # File uploader for multiple PDFs
        # uploaded_files = st.file_uploader(
        #     "Upload PDF(s)", type="pdf", accept_multiple_files=True
        # )

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
                    st.sidebar.write("Embeddings loaded successfully :partying_face:")
                except Exception as e:
                    st.error(f"Error loading embeddings: {str(e)}")
                    st.session_state.vectors = None

        # Process uploaded PDFs when the button is clicked
        if uploaded_files:
            if st.button("Process Documents"):
                with st.spinner("Processing documents... Please wait."):

                    def vector_embedding(uploaded_files):
                        if "vectors" not in st.session_state:
                            # Initialize embeddings if not already done
                            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
                                model="models/embedding-001"
                            )
                            all_docs = []

                            # Process each uploaded file
                            for uploaded_file in uploaded_files:
                                # Save the uploaded file temporarily
                                with tempfile.NamedTemporaryFile(
                                    delete=False, suffix=".pdf"
                                ) as temp_file:
                                    temp_file.write(uploaded_file.read())
                                    temp_file_path = temp_file.name

                                # Load the PDF document
                                loader = PyPDFLoader(temp_file_path)
                                docs = loader.load()  # Load document content

                                # Remove the temporary file
                                os.remove(temp_file_path)

                                # Add loaded documents to the list
                                all_docs.extend(docs)

                            # Split documents into manageable chunks
                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=1000, chunk_overlap=200
                            )
                            final_documents = text_splitter.split_documents(all_docs)

                            # Create a vector store with FAISS
                            st.session_state.vectors = FAISS.from_documents(
                                final_documents, st.session_state.embeddings
                            )

                    vector_embedding(uploaded_files)
                    st.sidebar.write("Documents processed successfully :partying_face:")

    else:
        st.error("Please enter both API keys to proceed.")

# Main area for chat interface
st.title("Chat with PDF :speech_balloon:")

# Initialize session state for chat messages if not already done
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input field for user queries
if human_input := st.chat_input("Ask something about the document"):
    st.session_state.messages.append({"role": "user", "content": human_input})
    with st.chat_message("user"):
        st.markdown(human_input)

    if "vectors" in st.session_state and st.session_state.vectors is not None:
        # Create and configure the document chain and retriever
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Get response from the assistant
        response = retrieval_chain.invoke({"input": human_input})
        assistant_response = response["answer"]

        # Append and display assistant's response
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response}
        )
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

        # Display supporting information from documents
        with st.expander("Supporting Information"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
        # Prompt user to upload and process documents if no vectors are available
        assistant_response = (
            "Please upload and process documents before asking questions."
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response}
        )
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
