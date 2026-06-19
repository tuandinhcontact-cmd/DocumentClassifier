import numpy as np
from custom_models.logistic_regression import CustomLogisticRegression

class CustomLinearSVM:
    def __init__(self, lr=0.01, lambda_param=0.01, epochs=50):
        self.lr = lr
        self.lambda_param = lambda_param  # Tham số kiểm soát Regularization (l2)
        self.epochs = epochs
        self.w = None
        self.b = 0.0
        self.platt_model = None  # Mô hình Logistic Regression dùng để hiệu chuẩn Platt Scaling
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        
        # Chuyển đổi nhãn từ {0, 1} sang {-1, 1} phục vụ Hinge Loss
        y_svm = np.where(y <= 0, -1, 1)
        
        # 1. Huấn luyện bằng Vectorized Batch Gradient Descent (Hinge Loss) với Pegasos-style learning rate schedule
        for epoch in range(self.epochs):
            # Tính f(x)
            decision = X.dot(self.w) + self.b
            # Xác định các điểm dữ liệu vi phạm vùng phân giới (margin < 1)
            margin_violation = (y_svm * decision) < 1
            
            # Tính gradient của hàm loss
            if np.sum(margin_violation) > 0:
                y_violation = y_svm * margin_violation
                dw = self.lambda_param * self.w - (X.T.dot(y_violation) / n_samples)
                db = -np.sum(y_violation) / n_samples
            else:
                dw = self.lambda_param * self.w
                db = 0.0
                
            # Học phí giảm dần theo Pegasos-style schedule
            eta = self.lr / (1.0 + self.lambda_param * epoch)
                
            # Cập nhật trọng số bằng học phí điều chỉnh
            self.w -= eta * dw
            self.b -= eta * db
                    
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
