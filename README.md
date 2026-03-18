## 📋 Mô tả dự án

**HealthChat AI** là một trợ lý ảo y tế sử dụng công nghệ **RAG (Retrieval-Augmented Generation)** kết hợp với mô hình ngôn ngữ lớn **Gemini 2.5 Flash**. Ứng dụng giúp người dùng mô tả triệu chứng bằng tiếng Việt, phân tích đa chiều, đưa ra đánh giá, nguyên nhân, gợi ý chẩn đoán, lời khuyên chăm sóc và hướng dẫn thăm khám (bao gồm gợi ý bệnh viện phù hợp theo vị trí).

Ứng dụng hỗ trợ **chat liền mạch** (nhớ lịch sử), tự động **hỏi thêm thông tin** khi triệu chứng mơ hồ và **cảnh báo khẩn cấp** khi phát hiện dấu hiệu nguy hiểm tính mạng.

---

## ✨ Tính năng chính

- Giao diện web hiện đại với Streamlit (dark theme, avatar, responsive)
- Chatbot nhớ lịch sử cuộc trò chuyện
- Tự động hỏi thêm thông tin khi triệu chứng mơ hồ
- Cảnh báo đỏ khẩn cấp (đau ngực, khó thở, tê liệt, nôn máu…)
- Gợi ý bệnh viện linh hoạt theo vị trí người dùng (Cần Thơ, TP.HCM, Hà Nội…)
- Xử lý tiếng Việt tốt nhờ Underthesea + RAG
- Miễn trừ trách nhiệm y tế rõ ràng

---

## 🛠 Công nghệ sử dụng

- **Frontend**: Streamlit
- **Backend**: LangChain, ChromaDB (Vector Store)
- **Embedding**: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- **LLM**: Google Gemini 2.5 Flash
- **Xử lý tiếng Việt**: underthesea
- **Dữ liệu**: ViMedical-Disease.csv → rag_db
- **Python**: 3.10+

---

## 📁 Cấu trúc thư mục
SANGLOCTRIEUCHUNG/
├── app.py                  # Ứng dụng chính (Streamlit)
├── build_rag.py            # Script xây dựng vector database
├── requirements.txt        # Danh sách thư viện
├── .streamlit/
│   └── secrets.toml        # (Gitignore) Chứa GOOGLE_API_KEY
├── rag_db/                 # (Gitignore) Vector database Chroma
├── ViMedical-Disease.csv   # Dữ liệu y khoa gốc
├── README.md
├── .gitignore
└── .venv/                  # (Gitignore) Môi trường ảo
## 🚀 Hướng dẫn cài đặt & chạy
### 1. Clone dự án
git clone <https://github.com/duong-vy/Sangloctrieuchung>
cd SANGLOCTRIEUCHUNG
2. Cài đặt môi trường
python -m venv .venv
.venv\Scripts\activate         ( Windows)
 source .venv/bin/activate     ( Mac/Linux)
pip install -r requirements.txt
3. Cấu hình API Key
Tạo thư mục .streamlit và file secrets.toml:
tomlGOOGLE_API_KEY = ""
4. Xây dựng cơ sở dữ liệu RAG (chạy 1 lần)
python build_rag.py
5. Chạy ứng dụng
streamlit run app.py

⚠️ Lưu ý quan trọng

Đây chỉ là công cụ hỗ trợ sàng lọc ban đầu, KHÔNG thay thế bác sĩ.
Người dùng phải đến cơ sở y tế để được chẩn đoán và điều trị chính xác.
Ứng dụng không lưu trữ thông tin cá nhân.


GitHub: [https://github.com/duong-vy/Sangloctrieuchung]
Demo: http://localhost:8501

Cảm ơn quý thầy cô và hội đồng đã xem xét đồ án!

