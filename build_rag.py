import os
import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def build_database():
    
    loader = CSVLoader(file_path="ViMedical_Disease.csv", encoding="utf-8")
    documents = loader.load()
    
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    
    batch_size = 4000
    if os.path.exists("./rag_db"):
        import shutil
        shutil.rmtree("./rag_db") 
    
    print(f"--- Đang nạp {len(documents)} dòng dữ liệu vào rag_db ---")
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        Chroma.from_documents(
            documents=batch, 
            embedding=embeddings, 
            persist_directory="./rag_db"
        )
        print(f"✅ Đã nạp xong {i + len(batch)} dòng...")

    print("🚀 HOÀN THÀNH: Đã tạo thư mục rag_db thành công!")

if __name__ == "__main__":
    build_database()