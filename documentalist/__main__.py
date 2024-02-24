import os
from langchain_community.llms import Ollama


def main():
    llm = Ollama(base_url=os.getenv('OLLAMA_BASE_URL'), model=os.getenv('OLLAMA_MODEL'))
    print(llm.invoke("Tell me a joke"))


if __name__ == "__main__":
    main()
