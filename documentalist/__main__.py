import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma


def main():
    print("Instantiating embeddings...")
    embeddings = OllamaEmbeddings(
        base_url=os.getenv('OLLAMA_BASE_URL'),
        model=os.getenv('OLLAMA_MODEL')
    )
    print("Instantiating the vectorstore...")
    chromadb = Chroma(
        persist_directory="./data/chromadb",
        embedding_function=embeddings,
        collection_name="documentalist"
    )
    print("Checking the vectorstore...")
    collection = chromadb.get()

    if len(collection['ids']) == 0:
        print("Instantiating document loader...")
        loader = DirectoryLoader(
            './data/dataset',
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            use_multithreading=True
        )
        print("Loading documents from the dataset...")
        documents = loader.load()

        print("Initializing the vectorstore...")
        chromadb = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory="./data/chromadb",
            collection_name="documentalist"
        )
        chromadb.persist()

    prompt = "What is the name of the algorithm described in the paper called 'Intriguingly Simple and Fast Transit Routing'?"

    print('Similarity search:')
    print(chromadb.similarity_search(prompt))

    print('Similarity search with score:')
    print(chromadb.similarity_search_with_score(prompt))

    print('Instantiating the LLM...')
    llm = Ollama(base_url=os.getenv('OLLAMA_BASE_URL'), model=os.getenv('OLLAMA_MODEL'))
    print('LLM search:')
    print(llm.invoke(prompt))


if __name__ == "__main__":
    main()
