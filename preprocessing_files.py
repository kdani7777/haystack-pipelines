from haystack.components.writers import DocumentWriter
from haystack.components.converters import MarkdownToDocument, PyPDFToDocument, TextFileToDocument
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore

"""
Initialize all the different file converters and the document joiner.
"""
document_store = InMemoryDocumentStore()
file_type_router = FileTypeRouter(mime_types=["text/plain", "application/pdf", "text/markdown"])
text_file_converter = TextFileToDocument()
markdown_converter = MarkdownToDocument()
pdf_converter = PyPDFToDocument()
document_joiner = DocumentJoiner()

"""
Document cleaner removes whitespace. Document splitter breaks them into chunks
of 150 words, with a bit of overlap to avoid missing context.
"""
document_cleaner = DocumentCleaner()
document_splitter = DocumentSplitter(split_by="word", split_length=150, split_overlap=50)

"""
Create embeddings from the documents and write them to the document store.
"""
document_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
document_writer = DocumentWriter(document_store)

"""
Add all the components to the indexing pipeline.
"""
preprocessing_pipeline = Pipeline()
preprocessing_pipeline.add_component(instance=file_type_router, name="file_type_router")
preprocessing_pipeline.add_component(instance=text_file_converter, name="text_file_converter")
preprocessing_pipeline.add_component(instance=markdown_converter, name="markdown_converter")
preprocessing_pipeline.add_component(instance=pdf_converter, name="pypdf_converter")
preprocessing_pipeline.add_component(instance=document_joiner, name="document_joiner")
preprocessing_pipeline.add_component(instance=document_cleaner, name="document_cleaner")
preprocessing_pipeline.add_component(instance=document_splitter, name="document_splitter")
preprocessing_pipeline.add_component(instance=document_embedder, name="document_embedder")
preprocessing_pipeline.add_component(instance=document_writer, name="document_writer")

"""
Connect the components.
"""
preprocessing_pipeline.connect("file_type_router.text/plain", "text_file_converter.sources")
preprocessing_pipeline.connect("file_type_router.application/pdf", "pypdf_converter.sources")
preprocessing_pipeline.connect("file_type_router.text/markdown", "markdown_converter.sources")
preprocessing_pipeline.connect("text_file_converter", "document_joiner")
preprocessing_pipeline.connect("pypdf_converter", "document_joiner")
preprocessing_pipeline.connect("markdown_converter", "document_joiner")
preprocessing_pipeline.connect("document_joiner", "document_cleaner")
preprocessing_pipeline.connect("document_cleaner", "document_splitter")
preprocessing_pipeline.connect("document_splitter", "document_embedder")
preprocessing_pipeline.connect("document_embedder", "document_writer")

"""
Run the preprocessing pipeline on local files.
*** These files do not exist in the project ***
"""
preprocessing_pipeline.run(
    {
        "file_type_router": {
            "sources": [
                "/content/vegan_sunflower_hemp_cheese_recipe.txt",
                "/content/vegan_keto_eggplant_recipe.pdf",
                "/content/vegan_flan_recipe.md",
            ]
        }
    }
)