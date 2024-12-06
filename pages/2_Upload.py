import streamlit as st
import tempfile
import pandas as pd
from langchain.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
import os

# Directory to store Chroma's database
PERSIST_DIR = "chroma_db"

def process_documents(files):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    all_docs = []
    
    for f in files:
        file_name = f.name.lower()
        documents = []

        if file_name.endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            # Update metadata to use original filename
            for doc in documents:
                doc.metadata["source"] = f.name
            os.unlink(tmp_path)  # Clean up temp file

        elif file_name.endswith(".csv"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            loader = CSVLoader(file_path=tmp_path)
            documents = loader.load()
            # Update metadata to use original filename
            for doc in documents:
                doc.metadata["source"] = f.name
            os.unlink(tmp_path)  # Clean up temp file

        elif file_name.endswith(".xlsx"):
            f.seek(0)
            df_dict = pd.read_excel(f, sheet_name=None)
            for sheet_name, df_sheet in df_dict.items():
                text = df_sheet.to_csv(index=False)
                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"source": f.name, "sheet_name": sheet_name}
                        )
                    )
        
        if documents:
            docs = text_splitter.split_documents(documents)
            all_docs.extend(docs)

    return all_docs

def get_existing_documents():
    if os.path.exists(PERSIST_DIR):
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        vectorstore = Chroma(
            collection_name="my_collection",
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR
        )
        # Get all documents from the collection
        docs = vectorstore.get()
        if docs and 'metadatas' in docs:
            # Extract unique source files with their IDs
            sources = {}
            for i, metadata in enumerate(docs['metadatas']):
                if 'source' in metadata:
                    sources[metadata['source']] = docs['ids'][i]
            return sources
    return {}

def delete_document(file_name):
    if os.path.exists(PERSIST_DIR):
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        vectorstore = Chroma(
            collection_name="my_collection",
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR
        )
        # Get all documents
        docs = vectorstore.get()
        if docs and 'metadatas' in docs:
            # Find all IDs associated with the file name
            ids_to_delete = [
                id for id, metadata in zip(docs['ids'], docs['metadatas'])
                if metadata.get('source') == file_name
            ]
            # Delete the documents
            if ids_to_delete:
                vectorstore.delete(ids_to_delete)
                vectorstore.persist()
                return True
    return False

st.title("📄 Upload Documents")

# Initialize session states
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# File uploader
uploaded_files = st.file_uploader(
    "Upload your documents",
    type=["pdf", "csv", "xlsx"],
    accept_multiple_files=True,
    help="Supported formats: PDF, CSV, XLSX"
)

# Display current uploads
if uploaded_files:
    st.write("### 📚 Current Uploads")
    for file in uploaded_files:
        st.write(f"- {file.name}")
    
    st.write(f"Total new documents: {len(uploaded_files)}")
    
    # Process button
    if st.button("Process Documents"):
        with st.spinner("Processing documents... This may take a while depending on file size."):
            docs = process_documents(uploaded_files)
            
            embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
            
            # Initialize or load Chroma vectorstore
            vectorstore = Chroma(
                collection_name="my_collection",
                embedding_function=embeddings,
                persist_directory=PERSIST_DIR
            )
            
            # Add documents in batches to avoid rate limits
            batch_size = 300
            docs_batches = [docs[i:i+batch_size] for i in range(0, len(docs), batch_size)]
            
            progress_bar = st.progress(0)
            for i, batch in enumerate(docs_batches):
                vectorstore.add_documents(batch)
                vectorstore.persist()
                progress = (i + 1) / len(docs_batches)
                progress_bar.progress(progress)
                st.write(f"Processed batch {i+1}/{len(docs_batches)}")
            
            # Initialize conversation memory
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
            conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                memory=memory,
            )
            
            # Store in session_state
            st.session_state.conversation_chain = conversation_chain
            st.session_state.vectorstore = vectorstore
            st.session_state.uploaded_files = uploaded_files
            
            st.success("✅ Documents processed successfully! You can now go to the Chat page to ask questions.")
            st.rerun()  # Updated from experimental_rerun to rerun
else:
    st.info("👆 Upload your documents to get started!")

# Display processed documents in a table
st.write("### 📚 Processed Documents")
existing_docs = get_existing_documents()
if existing_docs:
    # Create a DataFrame for the table
    data = []
    for doc_name in existing_docs.keys():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"📄 {doc_name}")
        with col2:
            if st.button("🗑️ Delete", key=f"delete_{doc_name}"):
                if delete_document(doc_name):
                    st.success(f"Deleted {doc_name}")
                    st.rerun()  # Updated from experimental_rerun to rerun
                else:
                    st.error(f"Failed to delete {doc_name}")
else:
    st.info("No documents have been processed yet.")
