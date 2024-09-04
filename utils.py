import re
from langchain_huggingface import HuggingFaceEmbeddings
import openai
from prompts import *
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# configuration for the apis
 OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#OPENAI_API_KEY = "sk-proj-"

if not OPENAI_API_KEY:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")


TOKENIZERS_PARALLELISM=False

chunk_size = 500
chunk_overlap = 100

# define the splitter 
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", ".", " ", ""]
)

# function to process and obtain chunks of document
def get_chunks(file):
    """
    Reads the file, split the document and return chunks of document
    """
    loader = PyPDFLoader(file)
    documents = loader.load()

    chunks = r_splitter.split_documents(documents)
    return chunks

if not os.path.exists('model'):
    os.mkdir('model')
    embeddings_model = SentenceTransformer('all-mpnet-base-v2')
    embeddings_model.save(path='model/all-mpnet-base-v2')

# get the embeddings model
model_name = "model/all-mpnet-base-v2"
model_kwargs = {}
encode_kwargs = {}
embeddings_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


# func to process and get a vectore db
def get_vectordb(file):
    """
    Process the given file and returns a vectore database
    """

    # reads, splits and obtain chunks of input file
    chunks = get_chunks(file)

    # embeds and indexing of chunks
    persist_directory = 'chroma/'
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        # persist_directory=persist_directory
    )

    return vector_store

def get_rel_docs(user_query, document_storage, filters=[]):
    """ 
    Gets the Relevant Docs Chunks from the VectorStore
    """
    rel_docs = document_storage.similarity_search(
        query=user_query,
        k=4,
        filter=filters, # [{"term": {"metadata.source.keyword": ""}}]
    )

    return rel_docs
    

def get_openai_response(prompt):
    #openai.api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error getting response from OpenAI API: {e}")
    
def get_query_response(user_query, document_storage):
    """
    Extract relevant context and Obtain the LLM response for given query.
    """
    # get the rel docs
    rel_docs = get_rel_docs(user_query, document_storage, filters=None)

    # prepare context
    context = ""
    for doc in rel_docs:
        context += "```\n" + "Source: " + doc.metadata['source'] + "\ncontent:\n" + doc.page_content + "```\n"

    # format the prompt with user query and context
    formatted_prompt = qna_prompt.format(user_query=user_query, context=context)
    
    # get openai response
    response = get_openai_response(formatted_prompt)

    return response, rel_docs

# function to process given documents and get llm responses to a list of questions
def get_responses_questions_list(filepath, queries_list):
    """
    Process the given file and generate llm responses.
    Returns: list of question-answer pair
    """
    responses = []
    # process the document and create a vectorstore 
    print("Processing the Document....")
    vector_store = get_vectordb(filepath)
    print("Document Processed and Vectore Database Created")
    print("Getting Responses for each question...")
    for query in queries_list:
        # get the llm response
        response, rel_docs = get_query_response(query, document_storage=vector_store)

        # store the question-answer pair in responses
        responses.append({
            "question": query,
            "answer": response
        })
    print("Generated the answers for all questions.")

    return responses

