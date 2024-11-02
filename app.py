from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.runnables import RunnableWithMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from gtts import gTTS
from io import BytesIO

session_store = {}

load_dotenv()

# Setting Environment Variables
collection_name = "AclGPT"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
PDF_PATH = os.getenv("PDF_path")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

st.set_page_config(page_title="ACL GPT", page_icon="🦜")
st.title("ACLBot: Your Resource for ACL Health and Recovery")

if "qdrant_initialized" not in st.session_state:
    st.session_state.qdrant_initialized = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "bot_audio" not in st.session_state:
    st.session_state.bot_audio = []

# Initialize memory globally
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
memory = st.session_state.memory

def process_document():
    pdf_files = [f for f in os.listdir(PDF_PATH) if f.endswith('.pdf')]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    doc_list = []
    for pdf_file in pdf_files:
        loader = PyPDFDirectoryLoader(PDF_PATH)
        pages = [p for p in loader.load() if os.path.basename(p.metadata['source']) == pdf_file]
        if not pages:
            print(f"No pages found for {pdf_file}")
            continue
        for page in pages:
            pg_splt = text_splitter.split_text(page.page_content)
            for pg_sub_splt in pg_splt:
                metadata = {
                    "book_name": pdf_file,
                    "page_no": page.metadata["page"] + 1,
                    "total_pages": len(pages)
                }
                doc_string = Document(page_content=pg_sub_splt, metadata=metadata)
                doc_list.append(doc_string)
    return doc_list

def create_qdrant():
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    if not client.collection_exists(collection_name):
        print("Creating new Qdrant collection")
        documents = process_document()
        qdrant = QdrantVectorStore.from_documents(
            documents=documents, 
            embedding=embeddings,
            collection_name=collection_name,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )
    else:
        print("Loading existing Qdrant collection")
        qdrant = QdrantVectorStore.from_existing_collection(
            collection_name=collection_name,
            embedding=embeddings,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )
    return qdrant

def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        source = doc.metadata.get("book_name", "unknown")
        page_no = doc.metadata.get("page_no", "unknown")  
        Total_pages = doc.metadata.get("total_pages", "unknown")
        formatted_doc = f"Source: {source}, Page: {page_no}\n\n{doc.page_content}, Total_pages: {Total_pages}"
        formatted_docs.append(formatted_doc)
    return "\n\n".join(formatted_docs)

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

def Retriver(input_query, llm):
    if not st.session_state.qdrant_initialized:
        documents = create_qdrant()
        st.session_state.qdrant = documents
        st.session_state.qdrant_initialized = True  
    qdrant = st.session_state.qdrant  
    memory = st.session_state.memory  
    prompt_str = """
You are an expert in ACL (Anterior Cruciate Ligament) medical issues.
For the first interaction, greet the user warmly and professionally.
Answer each question in detail, drawing exclusively from the provided context and previous conversations.
Use clear and simple language to ensure that non-technical users can understand the response.
Focus on comprehensively analyzing the context to formulate well-rounded answers, and verify that each response is fully supported by the provided material.
Always include accurate source and page number information, specifying precisely where the information was obtained from.
Don't mention that you have been given any context.
Do not self create the source and page number information.
If a question is outside the scope of the provided context, respond with: 
"I don't have enough information to answer that based on the provided context. Please keep your query relevant to ACL." 
Avoid using any external knowledge beyond the given context.

Context: {context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_messages(
        [                                  
            ("system", prompt_str),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ]
    )   
    num_chunks = 2
    retriever = qdrant.as_retriever(search_type="mmr", search_kwargs={"k": num_chunks})
    setup_runnable = RunnableLambda(lambda inputs: {
        "question": inputs["question"],
        "context": format_docs(retriever.get_relevant_documents(inputs["question"])),
        "history": memory.buffer
    })
    rag_chain = RunnableWithMessageHistory(
        setup_runnable | prompt | llm,
        get_session_history=get_session_history,
        memory=memory,
        input_messages_key="question",
        history_messages_key="history",     
    )
    memory.save_context({"question": input_query}, {"answer": ""})
    response_stream = rag_chain.stream(
        {"question": input_query},
        config={"configurable": {"session_id": "abc123"}}
    )
    full_response = ""
    for chunk in response_stream:
        if chunk.content is not None:
            full_response += chunk.content
    memory.save_context({"question": input_query}, {"answer": full_response})
    return full_response 

def Mistral_response(GROQ_API_KEY):
    chat_respone = ChatGroq(
        model_name= "mixtral-8x7b-32768",
        temperature=0.1,
        api_key= GROQ_API_KEY,
    )
    return chat_respone

def Llama_response(GROQ_API_KEY):
    chat_respone = ChatGroq(
        model_name= "llama-3.1-70b-versatile",
        temperature=0.1,
        api_key= GROQ_API_KEY,
    )
    return chat_respone

def Gemma2_response(GROQ_API_KEY):
    chat_respone = ChatGroq(
        model_name= "gemma2-9b-it",
        temperature=0.1,
        api_key= GROQ_API_KEY,
    )
    return chat_respone

def text_to_audio(text):
    tts = gTTS(text=text, lang='en', slow=False, tld="us")
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)    
    audio_bytes.seek(0)
    return audio_bytes

def display_chats(llm):
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "audio" in message:
                st.audio(message["audio"], format="audio/mp3")
    if prompt := st.chat_input("What is your question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            with st.container():
                st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response_chunk in Retriver(prompt, llm):
                full_response += response_chunk
                message_placeholder.markdown(full_response + "▌")
                with st.container():
                    message_placeholder.markdown(full_response)
            audio_bytes = text_to_audio(full_response)
            st.audio(audio_bytes, format="audio/mp3")
        messages = st.session_state.messages
        messages.append(
            {"role": "assistant", "content": full_response, "audio": audio_bytes})

def main():
    Selected_LLM = st.sidebar.selectbox(
        "Select Your LLM", 
        options=["mixtral-8x7b-32768", "llama-3.1-70b-versatile", "gemma2-9b-it"], 
        key="llm_selector_unique"
    )
    if Selected_LLM in ["mixtral-8x7b-32768", "llama-3.1-70b-versatile", "gemma2-9b-it"]:
        GROQ_API_KEY = st.sidebar.text_input("Enter your Groq API Key", type="password", key="groq_api_key")
        if not GROQ_API_KEY:
            st.warning("Groq API Key is required for Mistral AI, Llama 3.1 and Gemma2.")
            return
        elif Selected_LLM == "mixtral-8x7b-32768":
            llm = Mistral_response(GROQ_API_KEY)
        elif Selected_LLM == "llama-3.1-70b-versatile":
            llm = Llama_response(GROQ_API_KEY)
        elif Selected_LLM == "gemma2-9b-it":
            llm = Gemma2_response(GROQ_API_KEY)
    else:
        st.error("No model selected.")
        return
    display_chats(llm)

if __name__ == "__main__":
    main()
