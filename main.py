import os
import pickle
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_google_genai import ChatGoogleGenerativeAI

# Load .env variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Streamlit UI
st.title("News Research Tool")
st.sidebar.title("Article URLs")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)


file_path = "faiss_store.pkl"
urls = []

# Input 3 URLs
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

# Button to process URLs
process_url_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()

if process_url_clicked:
    main_placeholder.text("Loading data from URLs...")

    data = []
    for url in urls:
        if url.strip():
            try:
                loader = WebBaseLoader(url)
                data.extend(loader.load())
            except Exception as e:
                st.error(f"❌ Failed to load {url}: {e}")

    if not data:
        st.warning("⚠️ No data was loaded. Please check the URLs.")
    else:
        # Text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        main_placeholder.text("Text splitting in progress...✅")

        docs = text_splitter.split_documents(data)

        # Embedding + FAISS vector index
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embedding_model)

        main_placeholder.text("Building embedding vector index...✅")

        # Save vector index to file
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)

        st.success("✅ Vector index created and saved successfully!")

query = main_placeholder.text_input("Questions : ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            retriever = vectorstore.as_retriever()
            qa_chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=retriever,
           # return_source_documents=True  # Optional: to get docs back
            )
            
            response = qa_chain({"question": query},return_only_outputs=True)
            st.header("Answer")
            st.subheader(response["answer"])
