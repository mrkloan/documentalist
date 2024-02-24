import os
from langchain_community.llms import Ollama


def chat():
    llm = Ollama(model=os.getenv('OLLAMA_MODEL'))
    print(llm.invoke("Tell me a joke"))
