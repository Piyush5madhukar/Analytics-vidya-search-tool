import google.generativeai as genai
import pandas as pd
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os
import streamlit as st

# Load environment variables (ensure both GOOGLE_API_KEY and GEMINI_API_KEY are set in your .env file)
load_dotenv()

# Get API keys from environment
google_api_key = os.getenv("GOOGLE_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Metrics for benchmarking
metrics = {
    "vector_store_creation_time": 0.0,
    "document_processing_time": 0.0,
}

# Function to read CSV and prepare data
def load_csv_data(csv_file_path):
    """Load data from a CSV file."""
    start_time = time.time()
    data = pd.read_csv(csv_file_path)
    combined_texts = (data['Course Title'] + " " + data['Course Description'] + " " + data['Course curricullum']).tolist()
    metrics["document_processing_time"] = time.time() - start_time
    return combined_texts

# Function to split text into manageable chunks
def get_text_chunks(texts):
    """Split texts into smaller chunks."""
    start_time = time.time()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_chunks = []
    for text in texts:
        chunks = text_splitter.split_text(text)
        all_chunks.extend(chunks)
    metrics["vector_store_creation_time"] = time.time() - start_time
    return all_chunks

# Function to create and save the vector store
def create_vector_store(text_chunks, output_dir="faiss_index"):
    """Generate vector embeddings and save to a FAISS vector database."""
    start_time = time.time()

    # Initialize Google Generative AI Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(api_key=google_api_key, model="models/embedding-001")

    # Create FAISS vector store
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    # Save FAISS vector store locally
    vector_store.save_local(output_dir)

    metrics["vector_store_creation_time"] += time.time() - start_time
    return vector_store

# Function to generate answers using Gemini and vector search
def get_top_k_results(user_query, k=3):
    """Generate detailed responses using Gemini with top-k documents from vector store."""
    try:
        # Load FAISS vector store
        embeddings = GoogleGenerativeAIEmbeddings(api_key=google_api_key, model="models/embedding-001")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        # Perform similarity search
        docs = vector_store.similarity_search(query=user_query, k=k)

        # Extract content from the top-k documents
        context = "\n".join([doc.page_content for doc in docs])
        
        # Generate a detailed response using Gemini LLM
        prompt = f"""
        You are a helpful assistant recommending list of relevant courses based on user queries.Answer precisely dont show off. Use the context below to answer:
        
        Context:
        {context}
        
        Question:
        {user_query}
        
        Provide a detailed recommendation:
        """
        
        # Configure Gemini with Gemini API key
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        
        return response.text, docs

    except Exception as e:
        return f"Error: {e}", []

# Streamlit App for Course Recommendation System
def main():
    st.set_page_config(page_title="Course Recommendation System", page_icon=":mortar_board:")
    st.title("Course Recommendation System")

    # Local CSV file path
    csv_file_path = "data.csv"  # Replace with your actual CSV file path

    # Load, process the CSV, and create vector store
    with st.spinner("Processing the CSV file..."):
        texts = load_csv_data(csv_file_path)
        text_chunks = get_text_chunks(texts)
        create_vector_store(text_chunks)
        st.success("CSV file processed and vector store created!")

    user_query = st.text_input("Enter your course-related query:")
    num_results = st.slider("Number of results to return", min_value=1, max_value=10, value=3)

    if user_query:
        with st.spinner("Fetching the best courses..."):
            answer, docs = get_top_k_results(user_query, k=num_results)
            st.write("### Recommended Courses:")
            st.write(answer)

            st.write("### Matching Courses:")
            for i, doc in enumerate(docs):
                st.write(f"**{i + 1}. {doc.metadata.get('Course Title', 'N/A')}**")
                st.write(doc.page_content)

    # Analytics Section
    st.sidebar.subheader("Analytics Overview")
    st.sidebar.write(f"**Query Length:** {len(user_query)} characters" if user_query else "No query entered.")

if __name__ == "__main__":
    main()
