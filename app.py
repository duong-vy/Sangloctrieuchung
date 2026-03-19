import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter


# ====================== CẤU HÌNH GIAO DIỆN ======================
st.set_page_config(page_title="HealthChat AI", page_icon="🩺", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stChatMessage { 
        border-radius: 20px !important; 
        padding: 16px !important; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.3) !important; 
    }
    .stChatInput { border-radius: 25px !important; padding: 12px !important; }
    .main-title { 
        font-size: 42px; 
        background: linear-gradient(90deg, #00BFFF, #1E88E5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004416.png", width=120)
    st.markdown("### 🏥 HealthChat AI")
    
    if "location" not in st.session_state:
        st.session_state.location = "Cần Thơ"
    location = st.text_input("📍 Vị trí của bạn (tỉnh/thành phố)", value=st.session_state.location)
    st.session_state.location = location
    
    if st.button("🗑️ Xóa cuộc trò chuyện"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Xin chào! Tôi là Trợ lý AI y tế. Bạn đang gặp triệu chứng gì? Hãy kể chi tiết nhé."}
        ]
        st.rerun()
    
    st.warning("⚠️ Không thay thế bác sĩ. Đi khám ngay nếu có dấu hiệu nguy hiểm!")

# ====================== HEADER ======================
st.markdown('<div class="main-title">🩺 Trợ Lý Ảo Sàng Lọc Triệu Chứng</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Phân tích đa chiều - Gợi ý chuyên khoa - Chăm sóc sức khỏe thông minh</div>', unsafe_allow_html=True)

# ====================== KIỂM TRA & LOAD MODELS ======================
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("❌ Thiếu GOOGLE_API_KEY trong secrets.toml")
    st.stop()

if not os.path.exists("./rag_db"):
    st.error("❌ Chưa có thư mục rag_db. Vui lòng tạo dữ liệu trước.")
    st.stop()

@st.cache_resource(show_spinner="Đang tải hệ thống AI...")
def load_models():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = Chroma(persist_directory="./rag_db", embedding_function=embeddings)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        temperature=0.3
    )
    return vectorstore, llm

try:
    vectorstore, llm = load_models()
except Exception as e:
    st.error(f"⚠️ Lỗi kết nối hệ thống: {e}")
    st.stop()

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ====================== PROMPT (HỎI THÊM NẾU MƠ HỒ + CẢNH BÁO NGHIÊM TRỌNG) ======================
prompt_template = """
Bạn là bác sĩ tư vấn trực tuyến tận tâm tại Việt Nam.

Vị trí người dùng: {location}
Lịch sử cuộc trò chuyện:
{chat_history}

Dữ liệu y khoa tham khảo: {context}
Triệu chứng/ngữ cảnh hiện tại: {input}

**QUY TẮC BẮT BUỘC:**
- Nếu triệu chứng mơ hồ, thiếu thông tin quan trọng (vị trí đau cụ thể, thời gian xuất hiện, mức độ đau, triệu chứng kèm theo, tiền sử bệnh, tuổi, giới tính...) → Hãy **lịch sự hỏi thêm** trước khi đưa ra kết luận.
- Nếu có dấu hiệu nguy hiểm tính mạng (đau ngực dữ dội, khó thở nặng, tê liệt, nôn máu, sốt cao kèm cứng cổ, mất ý thức, chóng mặt kèm yếu tay chân, đau bụng dữ dội ở phụ nữ mang thai...) → 
  **PHẢI ĐẶT NGAY ĐẦU TIÊN: 🚨 CẢNH BÁO KHẨN CẤP** và khuyên gọi **115 hoặc đến cấp cứu NGAY**.

Trả lời bằng giọng văn nhẹ nhàng, đồng cảm theo đúng cấu trúc sau:

**1. 🔍 Đánh giá triệu chứng:**
**2. 🦠 Nguyên nhân có thể:**
**3. 🩺 Gợi ý chẩn đoán sơ bộ:**
**4. 💡 Giải pháp & Lời khuyên chăm sóc:**
**5. 🏥 Hướng dẫn thăm khám & Bệnh viện gợi ý:**
   - Chuyên khoa cần khám
   - Bệnh viện cụ thể (ưu tiên gần {location}, nếu cần chuyên sâu thì gợi ý bệnh viện lớn toàn quốc: Bạch Mai, Chợ Rẫy, Vinmec, 108, Từ Dũ, Việt Đức...)

---
⚠️ Lưu ý: Đây chỉ là phân tích ban đầu từ AI. Hãy đến bệnh viện để được bác sĩ khám trực tiếp.
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": itemgetter("input") | retriever | format_docs,
        "input": itemgetter("input"),
        "chat_history": itemgetter("chat_history"),
        "location": itemgetter("location"),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ====================== GIAO DIỆN CHAT ======================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Xin chào! Tôi là Trợ lý AI y tế. Bạn đang gặp triệu chứng gì? Hãy kể chi tiết nhé."}
    ]

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    avatar = "🩺" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Xử lý input mới
if query := st.chat_input("VD: Tôi bị đau quặn bụng dưới bên phải kèm buồn nôn..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="👤"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="🩺"):
        with st.spinner("🧠 Đang phân tích triệu chứng..."):
            try:
                processed_query = query
                
                # Lấy lịch sử gần nhất (8 tin nhắn)
                chat_history_str = "\n".join(
                    [f"{msg['role'].capitalize()}: {msg['content']}" 
                     for msg in st.session_state.messages[-8:]]
                )
                
                answer = rag_chain.invoke({
                    "input": processed_query,
                    "chat_history": chat_history_str,
                    "location": st.session_state.location
                })
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error("Hệ thống đang quá tải hoặc lỗi kết nối. Vui lòng thử lại sau vài giây.")
                st.info(f"Chi tiết lỗi: {str(e)}")