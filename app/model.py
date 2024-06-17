# Imports
import os
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Global caches
urls_cache = None
vectorstore_cache = None
rag_chain_cache = None

# Function to get cached RAG chain, can return None
def get_cached_rag_chain():
    return rag_chain_cache

# Function to get RAG chain
def get_rag_chain(urls):
    print('Fetching RAG chain...')

    # Global variables
    global urls_cache
    global vectorstore_cache
    global rag_chain_cache

    # Return chain if already exists, or generate new
    if(rag_chain_cache):
        return rag_chain_cache
    else:
        print('Creating new RAG chain...')
        urls_cache = urls
        rag_chain_cache, vectorstore_cache = _create_rag_chain(urls)
        print('RAG chain created.')
        return rag_chain_cache
    
# Function to reset the app
def reset_rag_chain():
    print('Resetting RAG...')

    # Global variables
    global urls_cache
    global vectorstore_cache
    global rag_chain_cache

    # Clear vectorstore and reset cache variables
    if(vectorstore_cache):
        print('Clearing vector store...')
        vectorstore_cache.delete_collection()
    urls_cache = None
    vectorstore_cache = None
    rag_chain_cache = None
    print('RAG reset done.')
    
# Function to generate RAG chain
def _create_rag_chain(urls):

    # Get Google API key
    GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
    if (not GOOGLE_API_KEY):
        raise Exception("Missing environment variable: GOOGLE_API_KEY")

    # Setup LLM
    llm = ChatGoogleGenerativeAI(model=os.environ['GOOGLE_MODEL_NAME'], google_api_key=GOOGLE_API_KEY, temperature=1e-8)

    # Load content from URLs
    docs = _load_urls(urls)

    # Create chunks
    chunks = _split_data(docs, 2000, 300)

    # Index documents in the vectorstore and get retriever
    embedding = GoogleGenerativeAIEmbeddings(model=os.environ['GOOGLE_EMBEDDING_MODEL_NAME'])
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # RAG chain
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Return RAG chain and vectorstore
    return rag_chain, vectorstore

# Function to load content from URLs
def _load_urls(urls):
    html2text = Html2TextTransformer()
    loader = AsyncHtmlLoader(urls)
    raw_docs = loader.load()
    docs = html2text.transform_documents(raw_docs)
    return docs

# Function to split documents into chunks
def _split_data(docs_to_split, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    return splitter.split_documents(docs_to_split)

# Document formatter for context docs
def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Main function to test this out
import sys
if __name__ == '__main__':
    print("Testing RAG with some simple prompts")

    # Get RAG chain
    rag_chain, vectorstore = _create_rag_chain([
        'https://www.nature.com/articles/d41586-024-01544-0',
        'https://www.nature.com/articles/d41586-024-01442-5',
        'https://www.nature.com/articles/d41586-024-01314-y',
        'https://www.nature.com/articles/d41586-024-01029-0'
    ])

    # Test prompts
    print(rag_chain.invoke('How are graphics processing units (GPUs) contributing to the advancement of artificial intelligence?'))
    print(rag_chain.invoke('What are the advantages of using Tensor Processing Units (TPUs) for AI tasks compared to traditional CPUs?'))
    print(rag_chain.invoke('What are the key features and advantages of the Blackwell chip compared to its predecessor, the Hopper chip?'))

    # Exit with code 0
    sys.exit(0)
