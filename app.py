import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain.memory import ConversationBufferMemory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader  
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from gtts import gTTS
from io import BytesIO
import os
from dotenv import load_dotenv

# Initialize session state
@st.cache_resource
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True)
    if "qdrant_client" not in st.session_state:
        st.session_state.qdrant_client = None
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return st.session_state

# Cache PDF processing
@st.cache_data
def process_document(pdf_path):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    
    loader = PyPDFDirectoryLoader(pdf_path)
    documents = loader.load()
    doc_list = []
    
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        for chunk in chunks:
            metadata = {
                "book_name": os.path.basename(doc.metadata['source']),
                "page_no": doc.metadata["page"] + 1,
                "total_pages": len(documents)
            }
            doc_list.append(Document(page_content=chunk, metadata=metadata))
    
    return doc_list

# Cache Qdrant initialization
@st.cache_resource
def initialize_qdrant(qdrant_url, qdrant_api_key, collection_name):
    try:
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=10
        )
        
        # Only create collection if it doesn't exist
        if not client.collection_exists(collection_name):
            documents = process_document(os.getenv("PDF_path"))
            vector_store = QdrantVectorStore.from_documents(
                documents=documents,
                embedding=st.session_state.embeddings,
                collection_name=collection_name,
                url=qdrant_url,
                api_key=qdrant_api_key,
            )
        else:
            vector_store = QdrantVectorStore.from_existing_collection(
                collection_name=collection_name,
                embedding=st.session_state.embeddings,
                url=qdrant_url,
                api_key=qdrant_api_key,
            )
        
        return vector_store
    except Exception as e:
        st.error(f"Failed to initialize Qdrant: {str(e)}")
        return None

# Cache LLM initialization
@st.cache_resource
def initialize_llm(llm_type, api_key):
    if llm_type == "Google Gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            api_key=api_key,
            temperature=0.1,
            streaming=True
        )
    elif llm_type == "Mistral AI":
        return ChatGroq(
            model_name="mixtral-8x7b-32768",
            temperature=0.1,
            api_key=api_key,
        )
    elif llm_type == "Llama 3.1":
        return ChatGroq(
            model_name="llama-3.1-70b-versatile",
            temperature=0.1,
            api_key=api_key,
        )

# Optimize retriever function
def get_relevant_context(vector_store, query, num_chunks=2):
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": num_chunks}
    )
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join([
        f"Source: {doc.metadata.get('book_name', 'unknown')}, "
        f"Page: {doc.metadata.get('page_no', 'unknown')}\n\n"
        f"{doc.page_content}"
        for doc in docs
    ])

# Cache audio generation
@st.cache_data
def generate_audio(text):
    audio_bytes = BytesIO()
    tts = gTTS(text=text, lang='en', slow=False, tld="us")
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes

def main():
    st.set_page_config(page_title="ACL GPT", page_icon="🦜")
    st.title("ACLBot: Your Resource for ACL Health and Recovery")
    
    # Initialize session state
    session_state = init_session_state()
    
    # Sidebar settings
    with st.sidebar:
        llm_type = st.selectbox(
            "Select Your LLM",
            options=["Google Gemini", "Mistral AI", "Llama 3.1"],
            key="llm_selector"
        )
        
        api_key = st.text_input(
            "Enter API Key",
            type="password",
            key="api_key"
        )
        
        enable_audio = st.checkbox("Enable Audio Responses", value=False)

    if not api_key:
        st.warning(f"Please enter your API key for {llm_type}")
        return

    # Initialize Qdrant
    vector_store = initialize_qdrant(
        os.getenv("QDRANT_URL"),
        os.getenv("QDRANT_API_KEY"),
        "AclGPT"
    )
    
    if not vector_store:
        st.error("Failed to initialize vector store")
        return

    # Initialize LLM
    llm = initialize_llm(llm_type, api_key)
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "audio" in message and enable_audio:
                st.audio(message["audio"], format="audio/mp3")

    # Chat input
    if prompt := st.chat_input("What is your question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Get context
            context = get_relevant_context(vector_store, prompt)
            
            # Generate response
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are an expert in ACL medical issues. Answer questions using only the provided context."""),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}")
            ])
            
            response = ""
            for chunk in llm.stream(prompt_template.format_prompt(
                question=prompt,
                history=session_state.memory.buffer
            )):
                if chunk.content:
                    response += chunk.content
                    message_placeholder.markdown(response + "▌")
            
            message_placeholder.markdown(response)
            
            # Generate audio if enabled
            if enable_audio:
                audio_bytes = generate_audio(response)
                st.audio(audio_bytes, format="audio/mp3")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "audio": audio_bytes
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
            
            # Update memory
            session_state.memory.save_context(
                {"question": prompt},
                {"answer": response}
            )

if __name__ == "__main__":
    load_dotenv()
    main()
