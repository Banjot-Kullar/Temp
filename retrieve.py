from langchain import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain_community.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
import json

# Initialize LLM
local_llm = "./BioMistral-7B.Q4_K_M.gguf"
llm = LlamaCpp(
    model_path=local_llm,
    temperature=0.3,
    max_tokens=2048,
    top_p=1
)

print("LLM Initialized....")

# Define prompt template
prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:
"""

# Load embeddings
embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

# Connect to Qdrant
url = "http://localhost:6333"
client = QdrantClient(url=url, prefer_grpc=False)

# Initialize vector database
db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db")

# Prepare retrieval settings
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
retriever = db.as_retriever(search_kwargs={"k": 1})

# **Function to get response**
def get_response(query):
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True
    )
    
    response = qa(query)
    print(response)

    answer = response['result']
    source_document = response['source_documents'][0].page_content
    doc = response['source_documents'][0].metadata['source']
    
    response_data = {"answer": answer, "source_document": source_document, "doc": doc}
    
    return json.dumps(response_data, indent=4)

# **Example usage**
if __name__ == "__main__":
    query = input("Enter your query: ")
    result = get_response(query)
    print("\nResponse:\n", result)