import streamlit as st
import os
from utils import *

def main():
    st.title("PDF Q&A with OpenAI")

    # Get the OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")
        os.environ["OPENAI_API_KEY"] = openai_api_key

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

    # Question input
    question = st.text_input("Enter your question:")

    # Process the question if both file and question are provided
    if uploaded_file is not None and question:
        try:
            with st.spinner("Processing your document..."):
                # Process the uploaded file
                file_path = "uploaded_" + uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())
                # Get responses 
                responses = get_responses_questions_list(file_path, [question])
                st.write("**Answer:**", responses[0]['answer'])
                # Delete the temporary file (optional, for cleanup)
                os.remove(file_path)

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()