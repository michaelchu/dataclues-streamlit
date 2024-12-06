import streamlit as st

st.set_page_config(page_title="DataClues.ai", page_icon="🤖", layout="wide")

st.title("🤖 DataClues.ai")
st.markdown("""
Welcome to DataClues! This application allows you to:

1. 📄 Upload documents (PDF, CSV, XLSX)
2. 💬 Chat with an AI about your documents

To get started:
1. Click on "Upload" in the navigation menu
2. Upload your documents and process them
3. Go to "Chat" to start asking questions about your documents
""")

# Initialize session states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
