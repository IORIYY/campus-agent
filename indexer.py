import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

print("Step 1: loading docs...")
loader = DirectoryLoader(
    "docs",
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
documents = loader.load()
print(f"Loaded {len(documents)} documents")

print("Step 2: splitting...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

print("Step 3: loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print("Step 4: writing to Chroma...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print(f"Done. Saved to ./chroma_db, total {len(chunks)} chunks")