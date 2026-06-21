import numpy as np
from custom_models.logistic_regression import CustomLogisticRegression

class CustomLinearSVM:
    def __init__(self, lr=0.01, lambda_param=0.01, epochs=50, class_weight=None):
        self.lr = lr
        self.lambda_param = lambda_param  # Tham số kiểm soát Regularization (l2)
        self.epochs = epochs
        self.class_weight = class_weight
        self.w = None
        self.b = 0.0
        self.platt_model = None  # Mô hình Logistic Regression dùng để hiệu chuẩn Platt Scaling
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        
        # Chuyển đổi nhãn từ {0, 1} sang {-1, 1} phục vụ Hinge Loss
        y_svm = np.where(y <= 0, -1, 1)
        
        # Cân bằng nhãn (Class Weighting)
        sample_weights = np.ones(n_samples)
        if self.class_weight == 'balanced':
            n_pos = np.sum(y_svm == 1)
            n_neg = np.sum(y_svm == -1)
            weight_pos = n_samples / (2.0 * n_pos) if n_pos > 0 else 1.0
            weight_neg = n_samples / (2.0 * n_neg) if n_neg > 0 else 1.0
            sample_weights = np.where(y_svm == 1, weight_pos, weight_neg)
        
        # Adam Optimizer parameters
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        mw = np.zeros(n_features)
        vw = np.zeros(n_features)
        mb = 0.0
        vb = 0.0
        
        # 1. Huấn luyện bằng Vectorized Batch Gradient Descent (Hinge Loss) kết hợp Adam
        for epoch in range(1, self.epochs + 1):
            # Tính f(x)
            decision = X.dot(self.w) + self.b
            # Xác định các điểm dữ liệu vi phạm vùng phân giới (margin < 1)
            margin_violation = (y_svm * decision) < 1
            
            # Tính gradient của hàm loss
            if np.sum(margin_violation) > 0:
                weighted_y_violation = y_svm * margin_violation * sample_weights
                dw = self.lambda_param * self.w - (X.T.dot(weighted_y_violation) / n_samples)
                db = -np.sum(weighted_y_violation) / n_samples
            else:
                dw = self.lambda_param * self.w
                db = 0.0
                
            # Adam Update
            mw = beta1 * mw + (1 - beta1) * dw
            vw = beta2 * vw + (1 - beta2) * (dw ** 2)
            
            mb = beta1 * mb + (1 - beta1) * db
            vb = beta2 * vb + (1 - beta2) * (db ** 2)
            
            mw_hat = mw / (1 - beta1 ** epoch)
            vw_hat = vw / (1 - beta2 ** epoch)
            mb_hat = mb / (1 - beta1 ** epoch)
            vb_hat = vb / (1 - beta2 ** epoch)
            
            # Cập nhật trọng số
            self.w -= self.lr * mw_hat / (np.sqrt(vw_hat) + epsilon)
            self.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + epsilon)
                    
        # 2. Hiệu chuẩn Platt Scaling (Platt's probabilistic calibration)
        # Tính toán giá trị quyết định f(x) = w^T x + b cho toàn bộ tập Train
        decision_values = X.dot(self.w) + self.b
        
        # Huấn luyện một mô hình Logistic Regression 1D trên f(x)
        self.platt_model = CustomLogisticRegression(lr=0.1, epochs=100)
        
        # X của Platt Model là ma trận 1 cột chứa giá trị quyết định của SVM
        X_platt = decision_values.reshape(-1, 1)
        self.platt_model.fit(X_platt, y)
        
        return self
        
    def predict_proba(self, X):
        # Tính giá trị quyết định f(x)
        decision_values = X.dot(self.w) + self.b
        X_platt = decision_values.reshape(-1, 1)
        
        # Dự đoán xác suất bằng mô hình Platt hiệu chuẩn
        return self.platt_model.predict_proba(X_platt)
        
    def predict(self, X):
        return (X.dot(self.w) + self.b >= 0.0).astype(int)
