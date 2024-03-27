from haystack.document_stores.in_memory import InMemoryDocumentStore
from datasets import load_dataset
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
import os
from getpass import getpass
from haystack.components.generators import OpenAIGenerator
from haystack import Pipeline

# document store to index documents that the question answering system will use
document_store = InMemoryDocumentStore()

"""
Fetch dataset from the Hugging Face Hub. Convert data into Haystack Documents.
"""
dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]

"""
Store data in the document store with embeddings
and then download the embedding model.
"""
doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
doc_embedder.warm_up()

"""
Create embeddings for each document and save these embeddings in Document
object's embedding field. Write the Documents to the DocumentStore.
"""
docs_with_embeddings = doc_embedder.run(docs)
document_store.write_documents(docs_with_embeddings["documents"])

"""
Initialize a text embedder to create an embedding for the user query.
"""
text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

"""
Initialize the Retriever. This will get the relevant documents to query.
"""
retriever = InMemoryEmbeddingRetriever(document_store)

"""
Create a custom prompt for a generative question answering task using RAG.
"""
template = """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""
prompt_builder = PromptBuilder(template=template)

"""
Initialize Generator to communicate with OpenAI GPT model.
"""
os.environ["OPENAI_API_KEY"] = getpass("Enter OpenAI API key: ")
generator = OpenAIGenerator(model="gpt-3.5-turbo")

"""
Build the pipeline.

Connections:
1) text_embedder.embedding -> retriever.query_embedding
2) retriever -> prompt_builder.documents
3) prompt_builder -> llm
"""
basic_rag_pipeline = Pipeline()
# Add components to pipeline
basic_rag_pipeline.add_component("text_embedder", text_embedder)
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
basic_rag_pipeline.add_component("llm", generator)

# Now, connect the components to each other
basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
basic_rag_pipeline.connect("retriever", "prompt_builder.documents")
basic_rag_pipeline.connect("prompt_builder", "llm")


"""
Ask a question.
"""
example_questions = [
    "Where is Gardens of Babylon?",
    "Why did people build Great Pyramid of Giza?",
    "What does Rhodes Statue look like?",
    "Why did people visit the Temple of Artemis?",
    "What is the importance of Colossus of Rhodes?",
    "What happened to the Tomb of Mausolus?",
    "How did Colossus of Rhodes collapse?",
]

question = "What does Rhodes Statue look like?"

response = basic_rag_pipeline.run({"text_embedder": {"text": question}, "prompt_builder": {"question": question}})

print(response["llm"]["replies"][0])