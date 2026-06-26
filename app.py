import os
import pickle
import re
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Import các class cần thiết để unpickle flat_gridsearch_model.pkl
from custom_models.logistic_regression import CustomLogisticRegression
from custom_models.softmax_regression import CustomSoftmaxRegression
from custom_models.multinomial_nb import CustomMultinomialNB
from custom_models.linear_svm import CustomLinearSVM
from custom_models.flat_ensemble import CustomOneVsRestClassifier, FlatSoftVotingClassifier

app = Flask(__name__)

# ─── Load model ──────────────────────────────────────────
MODEL_PATH = "models/flat_gridsearch_model.pkl"
model_data = None

if os.path.exists(MODEL_PATH):
    print(f"Đang tải mô hình từ {MODEL_PATH}...")
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    print("✅ Tải mô hình thành công!")
else:
    print(f"⚠️  Không tìm thấy {MODEL_PATH}!")


# ─── Text cleaning ───────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    words = text.split()
    cleaned = [w for w in words if w not in ENGLISH_STOP_WORDS]
    return " ".join(cleaned)


# ─── Routes ──────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


@app.route('/predict', methods=['POST'])
def predict():
    if model_data is None:
        return jsonify({'error': 'Mô hình chưa được tải!'}), 500

    data = request.get_json()
    headline          = data.get('headline', '')
    short_description = data.get('short_description', '')

    combined = headline + " " + short_description
    cleaned  = clean_text(combined)

    if not cleaned.strip():
        return jsonify({'error': 'Nội dung đầu vào trống!'}), 400

    model      = model_data['model']
    vectorizer = model_data['vectorizer']
    X_vec      = vectorizer.transform([cleaned])

    # Predict
    final_class = model.predict(X_vec)[0]

    # Lấy xác suất từng model thành phần
    path = []
    for idx, (name, est) in enumerate(model.estimators):
        probs    = est.predict_proba(X_vec)[0]
        pred_idx = np.argmax(probs)

        # Lấy tên classes
        if hasattr(est, 'classes'):
            cls_arr = est.classes
        else:
            cls_arr = model.classes

        pred_class = cls_arr[pred_idx]
        confidence = float(probs[pred_idx])

        path.append({
            'stage':            idx + 1,
            'model_name':       name,
            'class_name':       pred_class,
            'probability':      confidence,
            'matched':          pred_class == final_class,
            'status':           f'Model {name}',
        })

    # Soft Voting result
    ensemble_probs = model.predict_proba(X_vec)[0]
    cls_arr = next((e.classes for _, e in model.estimators if hasattr(e, 'classes')), model.classes)
    final_idx = np.argmax(ensemble_probs)
    path.append({
        'stage':       len(model.estimators) + 1,
        'model_name':  'Soft Voting',
        'class_name':  final_class,
        'probability': float(ensemble_probs[final_idx]),
        'matched':     True,
        'status':      'Kết quả Soft Voting cuối cùng',
    })

    return jsonify({
        'headline':          headline,
        'short_description': short_description,
        'cleaned_text':      cleaned,
        'prediction':        final_class,
        'cascade_path':      path,
        'model_type':        'flat_soft_voting_gridsearch',
    })


@app.route('/model_info', methods=['GET'])
def model_info():
    """Endpoint trả về thông tin model và best params."""
    if model_data is None:
        return jsonify({'error': 'Chưa load model'}), 500
    return jsonify({
        'model_type':  'Flat Soft Voting Ensemble (GridSearchCV)',
        'best_params': model_data.get('best_params', {}),
        'classes':     model_data.get('classes', []),
        'metrics': {
            'test_accuracy':  '70.03%',
            'macro_f1':       '68.17%',
        }
    })


if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    print("Khởi chạy Flask server tại http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True)
