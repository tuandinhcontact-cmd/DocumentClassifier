import os
import pickle
import re
from flask import Flask, request, jsonify, send_from_directory
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Import các Class tùy chỉnh để tránh lỗi unpickle AttributeError
try:
    from train_cascade_ensemble import HierarchicalCascadeEnsembleClassifier
except ImportError:
    pass

try:
    from train_cascade import HierarchicalCascadeClassifier
except ImportError:
    pass

try:
    from train_custom_tfidf_cascade import CustomHierarchicalCascadeClassifier, CustomVotingClassifier
    from custom_models.logistic_regression import CustomLogisticRegression
    from custom_models.multinomial_nb import CustomMultinomialNB
    from custom_models.linear_svm import CustomLinearSVM
    from custom_models.random_forest import CustomRandomForest, DecisionNode
except ImportError:
    pass

try:
    from train_hybrid_cascade import HybridCascadeClassifier, document_vector
except ImportError:
    pass

try:
    from train_custom_hybrid_cascade import CustomHybridCascadeClassifier, csr_to_list_of_dicts, tfidf_weighted_document_vector
except ImportError:
    pass

try:
    from train_3step_cascade import ThreeStepCascadeClassifier, CustomOneVsRestClassifier, CustomMultiClassVotingClassifier
except ImportError:
    pass

app = Flask(__name__)

# Tải mô hình đã đóng gói
MODEL_PATH = "cascade_model.pkl"
model_data = None

if os.path.exists(MODEL_PATH):
    print("Đang tải mô hình từ file pickle...")
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
else:
    print(f"Lỗi: Không tìm thấy file mô hình tại {MODEL_PATH}!")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Giữ lại chữ cái thường và khoảng trắng
    text = re.sub(r'[^a-zA-Záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ\s]', ' ', text)
    words = text.split()
    # Loại bỏ stopword tiếng Anh
    cleaned_words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    return " ".join(cleaned_words)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/predict', methods=['POST'])
def predict():
    if model_data is None:
        return jsonify({'error': 'Mô hình chưa được tải thành công trên Server!'}), 500
        
    data = request.get_json()
    headline = data.get('headline', '')
    short_description = data.get('short_description', '')
    
    # 1. Gộp và làm sạch văn bản
    combined = headline + " " + short_description
    cleaned = clean_text(combined)
    
    if not cleaned.strip():
        return jsonify({'error': 'Nội dung đầu vào trống hoặc chỉ chứa ký tự đặc biệt!'}), 400
        
    cascade_clf = model_data['model']
    classes_order = model_data['classes_order']
    
    # Check if this is the hybrid model
    is_hybrid = False
    try:
        if ('HybridCascadeClassifier' in globals() and isinstance(cascade_clf, HybridCascadeClassifier)) or \
           ('CustomHybridCascadeClassifier' in globals() and isinstance(cascade_clf, CustomHybridCascadeClassifier)):
            is_hybrid = True
    except NameError:
        pass

    # Check if this is the 3-step cascade model
    is_3step = False
    try:
        if 'ThreeStepCascadeClassifier' in globals() and isinstance(cascade_clf, ThreeStepCascadeClassifier):
            is_3step = True
    except NameError:
        pass
        
    # 2. Vector hóa bằng TF-IDF (nếu là mô hình truyền thống)
    if not is_hybrid and not is_3step:
        vectorizer = model_data['vectorizer']
        X_vector = vectorizer.transform([cleaned])
    elif is_3step:
        vectorizer = model_data['vectorizer']
        X_vector = vectorizer.transform([cleaned])
        
        final_class = cascade_clf.predict(X_vector)[0]
        path = []
        
        # Step 1
        prob1 = cascade_clf.step1_model.predict_proba(X_vector)[0][1]
        if prob1 >= 0.5:
            path.append({
                'stage': 1, 'class_name': cascade_clf.target1, 'probability': float(prob1),
                'matched': True, 'status': 'Chấp nhận (Matched)',
                'feature_type': 'tfidf', 'class_size': 'N/A', 'training_samples': 'N/A'
            })
        else:
            path.append({
                'stage': 1, 'class_name': cascade_clf.target1, 'probability': float(prob1),
                'matched': False, 'status': 'Bỏ qua (Passed)',
                'feature_type': 'tfidf', 'class_size': 'N/A', 'training_samples': 'N/A'
            })
            
            # Step 2
            prob2 = cascade_clf.step2_model.predict_proba(X_vector)[0][1]
            if prob2 >= 0.5:
                path.append({
                    'stage': 2, 'class_name': cascade_clf.target2, 'probability': float(prob2),
                    'matched': True, 'status': 'Chấp nhận (Matched)',
                    'feature_type': 'tfidf', 'class_size': 'N/A', 'training_samples': 'N/A'
                })
            else:
                path.append({
                    'stage': 2, 'class_name': cascade_clf.target2, 'probability': float(prob2),
                    'matched': False, 'status': 'Bỏ qua (Passed)',
                    'feature_type': 'tfidf', 'class_size': 'N/A', 'training_samples': 'N/A'
                })
                
                # Step 3
                import numpy as np
                probs3 = cascade_clf.step3_model.predict_proba(X_vector)[0]
                max_prob3 = float(np.max(probs3))
                path.append({
                    'stage': 3, 'class_name': final_class, 'probability': max_prob3,
                    'matched': True, 'status': 'Chấp nhận đa lớp (Multiclass)',
                    'feature_type': 'tfidf', 'class_size': 'N/A', 'training_samples': 'N/A'
                })

        return jsonify({
            'headline': headline,
            'short_description': short_description,
            'cleaned_text': cleaned,
            'prediction': final_class,
            'cascade_path': path
        })
    
    path = []
    final_class = classes_order[-1]
    matched_index = -1
    
    # 3. Duyệt qua các model nhị phân
    for idx, class_name in enumerate(classes_order[:-1]):
        if class_name not in cascade_clf.models:
            continue
            
        # Trích xuất metadata cho Hybrid hoặc Cascade thường có lưu vết
        feature_type = 'tfidf'
        class_size = 'N/A'
        training_samples = 'N/A'
        
        if is_hybrid:
            metadata = cascade_clf.node_metadata.get(class_name, {})
            feature_type = metadata.get('feature_type', 'tfidf')
            class_size = metadata.get('class_size', 0)
            training_samples = metadata.get('training_samples', 0)
            
            # Dự đoán sử dụng phương thức dynamic feature extraction của Hybrid Cascade
            prob = cascade_clf.predict_proba_node([cleaned], class_name)[0][1]
        else:
            if hasattr(cascade_clf, 'node_metadata'):
                metadata = cascade_clf.node_metadata.get(class_name, {})
                class_size = metadata.get('class_size', 'N/A')
                training_samples = metadata.get('training_samples', 'N/A')
            model = cascade_clf.models[class_name]
            prob = model.predict_proba(X_vector)[0][1]
        
        # Nếu xác suất >= 0.5, bóc tách lớp này ra
        if prob >= 0.5:
            final_class = class_name
            matched_index = idx
            path.append({
                'stage': idx + 1,
                'class_name': class_name,
                'probability': float(prob),
                'matched': True,
                'status': 'Chấp nhận (Matched)',
                'feature_type': feature_type,
                'class_size': class_size,
                'training_samples': training_samples
            })
            break # Dừng chuỗi phân lớp
        else:
            path.append({
                'stage': idx + 1,
                'class_name': class_name,
                'probability': float(prob),
                'matched': False,
                'status': 'Bỏ qua (Passed)',
                'feature_type': feature_type,
                'class_size': class_size,
                'training_samples': training_samples
            })
            
    # Nếu đi đến cuối chuỗi mà không khớp lớp nào, mặc định thuộc về lớp cuối cùng
    if matched_index == -1:
        path.append({
            'stage': len(classes_order),
            'class_name': classes_order[-1],
            'probability': 1.0,
            'matched': True,
            'status': 'Lớp mặc định cuối chuỗi (Default)',
            'feature_type': 'N/A',
            'class_size': 'N/A',
            'training_samples': 'N/A'
        })
        
    return jsonify({
        'headline': headline,
        'short_description': short_description,
        'cleaned_text': cleaned,
        'prediction': final_class,
        'cascade_path': path
    })

if __name__ == '__main__':
    # Tạo thư mục static nếu chưa có
    os.makedirs('static', exist_ok=True)
    print("Khởi chạy Flask server tại http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True)
