
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

#filter required data
def filter_to_minimal_doc(docs : List[Document]) -> List[Document]:
    """
    Given a list of Document objs, return a new list of Document objs containing only 'source'
    in metadata and the original page content
    """
    minimal_doc: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_doc.append(
            Document(
                page_content = doc.page_content,
                metadata = {"source":src}
            )
        )
    return minimal_doc

#Extract text from PDF files
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents

#chunking - spilt doc into smaller chunk
def text_spilt(minimal_doc):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 20,
    )
    texts_chunk = text_splitter.split_documents(minimal_doc)
    return texts_chunk


def download_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceBgeEmbeddings(
        model_name = model_name
    )
    return embeddings
