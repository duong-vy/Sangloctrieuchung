import pandas as pd
from underthesea import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Mapping specialty 
specialty_map = {
    # Tim mạch
    "Bệnh Cơ Tim Giãn Nở": "Tim mạch",
    "Bệnh Cơ Tim Phì Đại": "Tim mạch",
    "Bệnh Cơ Tim Hạn Chế": "Tim mạch",
    "Bệnh Động Mạch Vành": "Tim mạch",
    "Bệnh Van Tim": "Tim mạch",
    "Bệnh Tim Bẩm Sinh": "Tim mạch",
    "Nhồi Máu Cơ Tim": "Tim mạch",
    "Suy Tim": "Tim mạch",
    "Rối Loạn Nhịp Tim": "Tim mạch",

    # Thần kinh
    "Alzheimer": "Thần kinh",
    "Động Kinh": "Thần kinh",
    "Đau Đầu Migraine": "Thần kinh",
    "Parkinson": "Thần kinh",
    "Tai Biến Mạch Máu Não": "Thần kinh",
    "Viêm Màng Não": "Thần kinh",

    # Tiêu hóa - Gan mật
    "Viêm Gan B": "Tiêu hóa - Gan mật",
    "Viêm Gan C": "Tiêu hóa - Gan mật",
    "Xơ Gan": "Tiêu hóa - Gan mật",
    "Viêm Dạ Dày": "Tiêu hóa - Gan mật",
    "Viêm Ruột Thừa": "Tiêu hóa - Gan mật",
    "Viêm Đại Tràng": "Tiêu hóa - Gan mật",
    "Loét Dạ Dày Tá Tràng": "Tiêu hóa - Gan mật",

    # Thận - Tiết niệu
    "Viêm Cầu Thận Lupus": "Thận - Tiết niệu",
    "Suy Thận": "Thận - Tiết niệu",
    "Sỏi Thận": "Thận - Tiết niệu",
    "Viêm Bàng Quang": "Thận - Tiết niệu",

    # Da liễu
    "Viêm Da Dị Ứng": "Da liễu",
    "Vảy Nến": "Da liễu",
    "Mụn Trứng Cá": "Da liễu",
    "Chàm": "Da liễu",

    # Cơ xương khớp
    "Thoát Vị Đĩa Đệm": "Cơ xương khớp",
    "Viêm Khớp Dạng Thấp": "Cơ xương khớp",
    "Loãng Xương": "Cơ xương khớp",
    "Gout": "Cơ xương khớp",

    # Hô hấp
    "Viêm Phổi": "Hô hấp",
    "Hen Suyễn": "Hô hấp",
    "Viêm Phế Quản": "Hô hấp",
    "Lao Phổi": "Hô hấp",

    # Nội tiết
    "Tiểu Đường": "Nội tiết",
    "Cường Giáp": "Nội tiết",
    "Suy Giáp": "Nội tiết",
    
}

df = pd.read_csv('ViMedical_Disease.csv', encoding='utf-8')
df['specialty'] = df['Disease'].map(specialty_map).fillna('Đa khoa')
# df = df[df['specialty'] != 'Đa khoa']  # Optional: lọc

def preprocess(text):
    return word_tokenize(str(text).lower(), format="text")

df['symptoms_clean'] = df['Question'].apply(preprocess)

X = df['symptoms_clean']
y = df['specialty']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1,3),
        max_features=8000
    )),
    ('clf', LogisticRegression(
        max_iter=2000,
        solver='lbfgs',
        class_weight='balanced'
    ))
])




pipeline.fit(X_train, y_train)

print("Accuracy:", pipeline.score(X_test, y_test))
print(classification_report(y_test, pipeline.predict(X_test)))

joblib.dump(pipeline, 'health_chatbot_model.pkl')
print("Model lưu thành công!")