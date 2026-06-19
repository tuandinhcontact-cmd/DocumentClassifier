#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "model_parameters.h"

#define PI 3.14159265358979323846

/* 1. Hàm kích hoạt Sigmoid */
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

/* 2. Hàm chuẩn hóa đặc trưng (StandardScaler) */
void scale_features(const double* raw_features, double* scaled_features) {
    for (int i = 0; i < NUM_FEATURES; i++) {
        scaled_features[i] = (raw_features[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
    }
}

/* 3. Dự đoán Logistic Regression */
double predict_logistic_regression(const double* scaled_features) {
    double z = LR_INTERCEPT;
    for (int i = 0; i < NUM_FEATURES; i++) {
        z += LR_COEF[i] * scaled_features[i];
    }
    return sigmoid(z);
}

/* 4. Dự đoán Gaussian Naive Bayes */
double predict_gaussian_nb(const double* scaled_features) {
    double log_likelihood[2] = {0.0, 0.0};
    
    for (int c = 0; c < 2; c++) {
        log_likelihood[c] = log(GNB_CLASS_PRIOR[c]);
        for (int i = 0; i < NUM_FEATURES; i++) {
            double mean = GNB_THETA[c][i];
            double variance = GNB_VAR[c][i];
            double diff = scaled_features[i] - mean;
            
            // Tính hàm mật độ xác suất phân phối chuẩn (Gaussian Log PDF)
            // Log PDF = -0.5 * log(2 * pi * variance) - (diff^2) / (2 * variance)
            double term1 = -0.5 * log(2 * PI * variance);
            double term2 = -(diff * diff) / (2 * variance);
            log_likelihood[c] += term1 + term2;
        }
    }
    
    // Softmax để chuyển log-likelihood thành xác suất thực tế
    // Dùng trick log-sum-exp để tránh tràn số (underflow/overflow)
    double max_log = log_likelihood[0] > log_likelihood[1] ? log_likelihood[0] : log_likelihood[1];
    double exp0 = exp(log_likelihood[0] - max_log);
    double exp1 = exp(log_likelihood[1] - max_log);
    
    return exp1 / (exp0 + exp1); // Trả về xác suất của lớp 1
}

/* 5. Dự đoán Support Vector Machine (SVC) với hiệu chuẩn Platt Scaling */
double predict_svc(const double* scaled_features) {
    double decision_value = SVM_INTERCEPT;
    for (int i = 0; i < NUM_FEATURES; i++) {
        decision_value += SVM_COEF[i] * scaled_features[i];
    }
    
    // Platt scaling formula: P = 1 / (1 + exp(A * decision_value + B))
    double platt_z = SVM_PROB_A * decision_value + SVM_PROB_B;
    return sigmoid(platt_z);
}

/* 6. Hàm đệ quy duyệt cây quyết định trong Random Forest */
double traverse_tree(const Node* tree, int node_idx, const double* scaled_features) {
    const Node* node = &tree[node_idx];
    
    // Nếu là lá (feature = -1), trả về giá trị xác suất lưu ở lá
    if (node->feature == -1) {
        return node->value;
    }
    
    // Duyệt nhánh trái hoặc phải dựa trên ngưỡng phân chia
    if (scaled_features[node->feature] <= node->threshold) {
        return traverse_tree(tree, node->left, scaled_features);
    } else {
        return traverse_tree(tree, node->right, scaled_features);
    }
}

/* 7. Dự đoán Random Forest */
double predict_random_forest(const double* scaled_features) {
    double total_prob = 0.0;
    for (int i = 0; i < RF_NUM_TREES; i++) {
        total_prob += traverse_tree(RF_TREES[i], 0, scaled_features);
    }
    return total_prob / RF_NUM_TREES; // Trung bình cộng xác suất của tất cả các cây
}

int main() {
    printf("=== C thuần: ML Models Inference & Soft Voting ===\n\n");
    
    // Mẫu dữ liệu test (Copy từ output của export_models.py)
    // Đây là mẫu Breast Cancer thứ 0 từ tập Test
    double test_features[30] = {
        12.47, 18.6, 81.09, 481.9, 0.09965, 0.1058, 0.08005, 0.03821, 0.1925, 0.06373, 
        0.3961, 1.044, 2.497, 30.29, 0.006953, 0.01911, 0.02701, 0.01037, 0.01782, 0.003586, 
        14.97, 24.64, 96.05, 677.9, 0.1426, 0.2378, 0.2671, 0.1015, 0.3014, 0.0875
    };
    
    printf("Mẫu dữ liệu đầu vào (đoạn đầu): %f, %f, %f, %f...\n\n", 
           test_features[0], test_features[1], test_features[2], test_features[3]);
    
    // Chuẩn hóa dữ liệu
    double scaled_features[NUM_FEATURES];
    scale_features(test_features, scaled_features);
    
    // 1. Logistic Regression
    double prob_lr = predict_logistic_regression(scaled_features);
    printf("1. Logistic Regression Prob: %.6f\n", prob_lr);
    
    // 2. Gaussian Naive Bayes
    double prob_gnb = predict_gaussian_nb(scaled_features);
    printf("2. Gaussian Naive Bayes Prob: %.6f\n", prob_gnb);
    
    // 3. Support Vector Machine (SVC)
    double prob_svc = predict_svc(scaled_features);
    printf("3. Support Vector Machine (SVC) Prob: %.6f\n", prob_svc);
    
    // 4. Random Forest
    double prob_rf = predict_random_forest(scaled_features);
    printf("4. Random Forest Prob: %.6f\n", prob_rf);
    
    // 5. Soft Voting Ensemble
    double prob_ensemble = (prob_lr + prob_gnb + prob_svc + prob_rf) / 4.0;
    printf("\n--> Soft Voting Ensemble Prob: %.6f\n", prob_ensemble);
    
    // Dự đoán nhãn
    int final_prediction = prob_ensemble >= 0.5 ? 1 : 0;
    printf("Dự đoán nhãn cuối cùng (Final Prediction): %d (%s)\n", 
           final_prediction, final_prediction == 1 ? "Benign - Lành tính" : "Malignant - Ác tính");
           
    return 0;
}
