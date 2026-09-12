import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

questions = [
    "转专业需要什么条件？",
    "毕业要修多少学分？",
    "学士学位怎么申请？"
]

for q in questions:
    print(f"\n{'='*50}")
    print(f"问题：{q}")
    results = vectorstore.similarity_search_with_score(q, k=2)
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n--- Top{i}（距离：{score:.4f}）---")
        print(f"来源：{doc.metadata.get('source', 'unknown')}")
        print(f"内容：{doc.page_content[:200]}...")