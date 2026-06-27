#!/bin/bash
# ============================================================
# run.sh — Script tiện ích để chạy các bước trong pipeline
# Sử dụng: bash run.sh <lệnh>
#
# Ví dụ:
#   bash run.sh merge        → Chạy merge_and_clean.py
#   bash run.sh convert      → Chạy convert.py
#   bash run.sh compare      → Chạy train_individual_models_comparison.py
#   bash run.sh ensemble     → Chạy train_flat_ensemble.py
#   bash run.sh gridsearch   → Chạy train_flat_gridsearch.py
#   bash run.sh train        → Chạy train_optimal_model_to_production.py
# ============================================================

PYTHON="/Users/gtuan/.gemini/antigravity/scratch/TExt-classifier/venv/bin/python"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Kiểm tra python
if [ ! -f "$PYTHON" ]; then
    echo "❌ Không tìm thấy Python tại: $PYTHON"
    exit 1
fi

cd "$SCRIPT_DIR"

case "$1" in
    convert)
        echo "🔄 Chạy convert.py (JSON → CSV)..."
        "$PYTHON" data/data_preparation/convert.py
        ;;
    merge)
        echo "🔄 Chạy merge_and_clean.py (Gộp 6 nguồn)..."
        "$PYTHON" data/data_preparation/merge_and_clean.py
        ;;
    compare)
        echo "🔄 Chạy train_individual_models_comparison.py..."
        "$PYTHON" training_scripts/train_individual_models_comparison.py
        ;;
    compare4k)
        echo "🔄 Chạy train_individual_models_comparison_4k.py..."
        "$PYTHON" training_scripts/train_individual_models_comparison_4k.py
        ;;
    compare12k)
        echo "🔄 Chạy train_individual_models_comparison_12k.py..."
        "$PYTHON" training_scripts/train_individual_models_comparison_12k.py
        ;;
    pilot_tfidf)
        echo "🔄 Chạy pilot_tfidf_softmax_4k.py..."
        "$PYTHON" training_scripts/pilot_tfidf_softmax_4k.py
        ;;
    pilot_w2v)
        echo "🔄 Chạy pilot_w2v_softmax_4k.py..."
        "$PYTHON" training_scripts/pilot_w2v_softmax_4k.py
        ;;
    ensemble)
        echo "🔄 Chạy train_flat_ensemble.py..."
        "$PYTHON" training_scripts/train_flat_ensemble.py
        ;;
    gridsearch)
        echo "🔄 Chạy train_flat_gridsearch.py..."
        "$PYTHON" training_scripts/train_flat_gridsearch.py
        ;;
    train)
        echo "🔄 Chạy train_optimal_model_to_production.py..."
        "$PYTHON" training_scripts/train_optimal_model_to_production.py
        ;;
    *)
        echo ""
        echo "📋 Cách dùng: bash run.sh <lệnh>"
        echo ""
        echo "  Bước 1 — Chuẩn bị dữ liệu:"
        echo "    bash run.sh convert     → Chuyển JSON → CSV (HuffPost)"
        echo "    bash run.sh merge       → Gộp 6 nguồn → data/merged_dataset.csv"
        echo ""
        echo "  Bước 2 — Huấn luyện mô hình:"
        echo "    bash run.sh pilot_tfidf → Thử nghiệm TF-IDF + Softmax (Cap 4k)"
        echo "    bash run.sh pilot_w2v   → Thử nghiệm Word2Vec + Softmax (Cap 4k)"
        echo "    bash run.sh compare     → So sánh từng mô hình (Cap 20k/nhãn)"
        echo "    bash run.sh compare4k   → So sánh từng mô hình (Cap 4k/nhãn)"
        echo "    bash run.sh compare12k  → So sánh từng mô hình (Cap 12k/nhãn)"
        echo "    bash run.sh ensemble    → Huấn luyện Soft Voting Ensemble"
        echo "    bash run.sh gridsearch  → Grid Search tối ưu hyperparameter"
        echo "    bash run.sh train       → Xuất mô hình tốt nhất ra production"
        echo ""
        ;;
esac
