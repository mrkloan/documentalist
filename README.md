# Documentalist

> RAG CLI built upon [🦜🔗LangChain](https://python.langchain.com) and [🦙Ollama](https://ollama.com/) to chat with your
> private PDF documents.

## 🚀 Quick Start

1. Prepare your dataset: put all the PDF documents used for training in the [`./data/dataset`](./data/dataset)
   directory.
2. Start the Ollama container: `docker compose up -d`
   (you can update the `OLLAMA_MODEL` environment variable defined in [`.env`](.env) in order to pull the model of your
   choice).
3. Create the embeddings and query your custom LLM: `poetry install && poetry run main`