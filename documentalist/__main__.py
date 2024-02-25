import click
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

PROMPT_TEMPLATE = """\
    You are an expert problem-solver, tasked with answering any question you might be asked.
    
    Generate a comprehensive and informative answer of 80 words or less for the \
    given question based solely on the provided context (document name and content). \
    You must only use information from the provided context. Use an unbiased and \
    journalistic tone. Combine context results together into a coherent answer. Do not \
    repeat text. Cite the most relevant contexts that answer the question accurately. \
    You should use bullet points in your answer for readability. Put citations where they apply \
    rather than putting them all at the end. 
    
    If there is nothing in the context relevant to the question at hand, just say "Hmm, \
    I'm not sure." Don't try to make up an answer. \
    
    Anything between the following `context` html blocks is retrieved from a knowledge \
    bank, not part of the conversation with the user. 
    
    <context>
        {context} 
    <context/>
    
    REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
    not sure." Don't try to make up an answer. Anything between the preceding 'context' \
    html blocks is retrieved from a knowledge bank, not part of the conversation with the \
    user.
    
    Question: {question}
    
    Helpful Answer:"""


@click.group()
def cli():
    """Train a custom RAG on your private PDF documents."""
    pass


@cli.command()
@click.option('--base-url', default=os.getenv('OLLAMA_BASE_URL'), help='The base URL of the Ollama server.')
@click.option('--model', default=os.getenv('OLLAMA_MODEL'), help='The name of the Ollama model to use.')
@click.option('--chromadb-path', default=os.getenv('CHROMADB_PATH'), help='The path to the persisted ChromaDB data.')
@click.option('--collection', default=os.getenv('CHROMADB_COLLECTION'), help='The newly trained collection.')
@click.option('--dataset-path', default=os.getenv('DATASET_PATH'), help='The path to the training dataset.')
def train(base_url, model, chromadb_path, collection, dataset_path):
    """Train a custom RAG on your private PDF documents."""
    click.echo("base_url: {:s}".format(base_url))
    click.echo("model: {:s}".format(model))
    click.echo("chromadb_path: {:s}".format(chromadb_path))
    click.echo("collection: {:s}".format(collection))
    click.echo("dataset_path: {:s}".format(dataset_path))

    click.echo("[1/5] Loading documents from the dataset...")
    loader = DirectoryLoader(
        dataset_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        use_multithreading=True
    )
    documents = loader.load()
    click.echo("\tNumber of loaded documents: {:d}".format(len(documents)))

    click.echo("[2/5] Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    splits = splitter.split_documents(documents)
    click.echo("\tNumber of split documents: {:d}".format(len(splits)))

    click.echo("[3/5] Creating embeddings and initializing the vectorstore...")
    embeddings = OllamaEmbeddings(
        base_url=base_url,
        model=model
    )
    chromadb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=chromadb_path,
        collection_name=collection
    )

    click.echo("[4/5] Persisting the vectorstore...")
    chromadb.persist()

    click.echo("[5/5] Done!")


@cli.command()
@click.option('--base-url', default=os.getenv('OLLAMA_BASE_URL'), help='The base URL of the Ollama server.')
@click.option('--model', default=os.getenv('OLLAMA_MODEL'), help='The name of the Ollama model to use.')
@click.option('--chromadb-path', default=os.getenv('CHROMADB_PATH'), help='The path to the persisted ChromaDB data.')
@click.option('--collection', default=os.getenv('CHROMADB_COLLECTION'), help='The collection containing the RAG.')
def chat(base_url, model, chromadb_path, collection):
    """Chat with your custom LLM."""
    click.echo("base_url: {:s}".format(base_url))
    click.echo("model: {:s}".format(model))
    click.echo("chromadb_path: {:s}".format(chromadb_path))
    click.echo("collection: {:s}".format(collection))

    embeddings = OllamaEmbeddings(
        base_url=base_url,
        model=model
    )
    chromadb = Chroma(
        persist_directory=chromadb_path,
        embedding_function=embeddings,
        collection_name=collection
    )
    retriever = chromadb.as_retriever()
    rag_prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = Ollama(
        base_url=base_url,
        model=model
    )
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
    )

    prompt = click.prompt("Question")
    answer = rag_chain.invoke(prompt)
    click.echo(answer.strip())


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    cli()
