import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os

st.title("💬 Chat with Your Documents")

# Set OpenAI API key from secrets
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm ready to help you with your documents. What would you like to know?"}]

# Initialize conversation memory in session state
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )

# Initialize or load the vectorstore from disk
PERSIST_DIR = "chroma_db"
if os.path.exists(PERSIST_DIR):
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()
    
    # Load the persisted vectorstore
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="my_collection"
    )
    
    # Configure retriever with improved search parameters
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Use MMR for better diversity in results
        search_kwargs={
            "k": 12,  # Increased number of documents to return
            "fetch_k": 30,  # Fetch more documents for MMR to choose from
            "lambda_mult": 0.5  # Lower lambda for more diversity (0.0-1.0)
        }
    )
    
    # Initialize the conversation chain with persisted memory
    llm = ChatOpenAI(temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory,
        return_source_documents=True,
        verbose=True
    )
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = conversation_chain({"question": prompt})
                msg = response["answer"]
                
                # Display only source filenames
                if "source_documents" in response:
                    sources = set()
                    for doc in response["source_documents"]:
                        if "source" in doc.metadata:
                            sources.add(doc.metadata["source"])
                    
                    if sources:
                        msg += "\n\nSources: " + ", ".join(sources)
                
                st.write(msg)
                
        st.session_state.messages.append({"role": "assistant", "content": msg})

    # Clear chat history button
    if st.button("Clear Chat History", key="clear_chat"):
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm ready to help you with your documents. What would you like to know?"}]
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        )
        st.rerun()
else:
    st.warning("⚠️ Please upload documents first! Go to the Upload Documents page to get started.")
