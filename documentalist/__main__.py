import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


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
    print("Checking the vectorstore content...")
    collection = chromadb.get()
    print("\tNumber of documents in the vectorstore: {:d}".format(len(collection['documents'])))

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
        print("\tNumber of loaded documents: {:d}".format(len(documents)))

        print("Instantiating document splitter...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
        print("Splitting documents...")
        splits = splitter.split_documents(documents)
        print("\tNumber of split documents: {:d}".format(len(splits)))

        print("Initializing the vectorstore...")
        chromadb = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./data/chromadb",
            collection_name="documentalist"
        )
        print("Persisting the vectorstore...")
        chromadb.persist()

    print('Instantiating the RAG chain...')
    retriever = chromadb.as_retriever()
    template = """\
    You are an expert problem-solver, tasked with answering any question you might be asked.\
    
    Generate a comprehensive and informative answer of 80 words or less for the \
    given question based solely on the provided context (document name and content). \
    You must only use information from the provided context. Use an unbiased and \
    journalistic tone. Combine context results together into a coherent answer. Do not \
    repeat text. Cite context documents using [${{number}}] notation. Only cite the most \
    relevant contexts that answer the question accurately. Place these citations at the end \
    of the sentence or paragraph that reference them - do not put them all at the end. If \
    different results refer to different entities within the same name, write separate \
    answers for each entity.
    
    You should use bullet points in your answer for readability. Put citations where they apply
    rather than putting them all at the end.
    
    If there is nothing in the context relevant to the question at hand, just say "Hmm, \
    I'm not sure." Don't try to make up an answer.
    
    Anything between the following `context`  html blocks is retrieved from a knowledge \
    bank, not part of the conversation with the user. 
    
    <context>
        {context} 
    <context/>
    
    REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
    not sure." Don't try to make up an answer. Anything between the preceding 'context' \
    html blocks is retrieved from a knowledge bank, not part of the conversation with the \
    user.\
    
    Question: {question}\
    
    Helpful Answer:"""
    rag_prompt = PromptTemplate.from_template(template)
    llm = Ollama(base_url=os.getenv('OLLAMA_BASE_URL'), model=os.getenv('OLLAMA_MODEL'))
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
    )

    print('Querying the LLM through the RAG chain...')
    prompt = "What is the name of the routing algorithm introduced in the paper 'Intriguingly Simple and Fast Transit Routing'?"
    answer = rag_chain.invoke(prompt)
    print("\tQuestion: {:s}".format(prompt))
    print("\tAnswer: {:s}".format(answer.strip()))


if __name__ == "__main__":
    main()
