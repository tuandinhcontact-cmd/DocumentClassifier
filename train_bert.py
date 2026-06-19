import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Thiết lập thiết bị chạy (Sử dụng Apple Silicon GPU 'mps' nếu có, hoặc CUDA, hoặc CPU)
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Đang sử dụng thiết bị: {device}")

def main():
    dataset_path = "cleaned_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"Lỗi: Không tìm thấy tệp {dataset_path}.")
        return

    # 1. Đọc dữ liệu sạch
    print("Đang đọc dữ liệu sạch...")
    df = pd.read_csv(dataset_path)
    df['cleaned_text'] = df['cleaned_text'].fillna('')
    
    # 2. Mã hóa nhãn (42 categories thành các số từ 0 đến 41)
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['category'])
    num_labels = len(label_encoder.classes_)
    
    # Lưu danh sách nhãn để sử dụng sau này
    print(f"Số lượng nhãn phân loại: {num_labels}")
    
    # Do Transformer huấn luyện rất nặng, bạn nên chạy trên toàn bộ dữ liệu khi có GPU.
    # Để demo chạy thử, chúng tôi lấy 20,000 mẫu. Xóa giới hạn này khi chạy trên GPU mạnh/Colab.
    df = df.sample(n=min(20000, len(df)), random_state=42).reset_index(drop=True)
    
    # 3. Chia tập train/test
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])
    
    # Chuyển đổi DataFrame sang HuggingFace Dataset
    train_dataset = Dataset.from_pandas(train_df[['cleaned_text', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['cleaned_text', 'label']])
    
    # 4. Tải Tokenizer của DistilBERT (phiên bản nhanh và nhẹ của BERT)
    print("Đang tải DistilBERT tokenizer...")
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples['cleaned_text'], padding="max_length", truncation=True, max_length=128)
    
    print("Đang mã hóa văn bản (Tokenizing)...")
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    val_tokenized = val_dataset.map(tokenize_function, batched=True)
    
    # 5. Tải mô hình DistilBERT tiền huấn luyện cho bài toán phân loại
    print("Đang tải mô hình DistilBERT...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)
    
    # 6. Cấu hình tham số huấn luyện (Training Arguments)
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir="./logs",
        logging_steps=100,
        report_to="none" # Tắt logging lên các dịch vụ bên ngoài
    )
    
    # Định nghĩa hàm tính toán chỉ số đánh giá
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        acc = (predictions == labels).mean()
        return {"accuracy": acc}
    
    import numpy as np
    
    # 7. Khởi tạo Trainer của HuggingFace
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        compute_metrics=compute_metrics,
    )
    
    # 8. Bắt đầu huấn luyện (Fine-tuning)
    print("Bắt đầu huấn luyện mô hình DistilBERT...")
    trainer.train()
    
    # Đánh giá trên tập Validation
    print("\nĐánh giá kết quả cuối cùng...")
    eval_results = trainer.evaluate()
    print(f"Độ chính xác (Accuracy) của DistilBERT: {eval_results['eval_accuracy']:.4%}")

if __name__ == "__main__":
    main()
