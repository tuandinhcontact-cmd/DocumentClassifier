import numpy as np

class CustomLogisticRegression:
    def __init__(self, lr=0.01, epochs=100, C=1.0, class_weight='balanced', solver='adam'):
        self.lr = lr
        self.epochs = epochs
        self.C = C  # Tham số kiểm soát điều chuẩn (Càng nhỏ L2 càng lớn)
        self.class_weight = class_weight
        self.solver = solver  # 'adam' hoặc 'gd'
        self.w = None
        self.b = 0.0
        
    def _sigmoid(self, z):
        # Giới hạn giá trị z để tránh tràn số exp
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
        
    def _compute_loss(self, X, y, w, b, sample_weights):
        n_samples = X.shape[0]
        z = X.dot(w) + b
        z_abs = np.abs(z)
        loss = np.log1p(np.exp(-z_abs)) + np.maximum(0, z) - y * z
        weighted_loss = np.sum(loss * sample_weights) / n_samples
        l2_reg = 0.5 * (1.0 / self.C) * np.sum(w ** 2)
        return weighted_loss + l2_reg
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        
        # 1. Tính toán trọng số cho mẫu nếu class_weight='balanced'
        if self.class_weight == 'balanced':
            y_series = np.array(y)
            classes, counts = np.unique(y_series, return_counts=True)
            class_weights = n_samples / (len(classes) * counts)
            weight_dict = dict(zip(classes, class_weights))
            sample_weights = np.array([weight_dict[val] for val in y_series])
        else:
            sample_weights = np.ones(n_samples)
            
        y_encoded = np.array(y)
        
        # 2. Huấn luyện bằng Adam hoặc Gradient Descent
        if self.solver == 'adam':
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            
            # Khởi tạo mô-men cho Adam
            m_w = np.zeros(n_features)
            v_w = np.zeros(n_features)
            m_b = 0.0
            v_b = 0.0
            
            for t in range(1, self.epochs + 1):
                z = X.dot(self.w) + self.b
                predictions = self._sigmoid(z)
                
                errors = predictions - y_encoded
                weighted_errors = errors * sample_weights
                
                dw = (X.T.dot(weighted_errors) / n_samples) + (1.0 / self.C) * self.w
                db = np.sum(weighted_errors) / n_samples
                
                # Cập nhật moving averages
                m_w = beta1 * m_w + (1.0 - beta1) * dw
                v_w = beta2 * v_w + (1.0 - beta2) * (dw ** 2)
                m_b = beta1 * m_b + (1.0 - beta1) * db
                v_b = beta2 * v_b + (1.0 - beta2) * (db ** 2)
                
                # Hiệu chuẩn bias correction
                m_w_hat = m_w / (1.0 - beta1 ** t)
                v_w_hat = v_w / (1.0 - beta2 ** t)
                m_b_hat = m_b / (1.0 - beta1 ** t)
                v_b_hat = v_b / (1.0 - beta2 ** t)
                
                # Cập nhật trọng số
                self.w -= (self.lr / (np.sqrt(v_w_hat) + eps)) * m_w_hat
                self.b -= (self.lr / (np.sqrt(v_b_hat) + eps)) * m_b_hat
                
        else: # Solver = 'gd' (Gradient Descent với Backtracking Line Search)
            for _ in range(self.epochs):
                z = X.dot(self.w) + self.b
                predictions = self._sigmoid(z)
                
                errors = predictions - y_encoded
                weighted_errors = errors * sample_weights
                
                dw = (X.T.dot(weighted_errors) / n_samples) + (1.0 / self.C) * self.w
                db = np.sum(weighted_errors) / n_samples
                
                # Tính toán tổn thất hiện tại
                current_loss = self._compute_loss(X, y_encoded, self.w, self.b, sample_weights)
                
                # Backtracking Line Search (Armijo condition)
                alpha = self.lr
                beta = 0.5  # Hệ số co hẹp học phí
                c = 1e-4    # Tham số kiểm tra độ dốc Armijo
                grad_sq_norm = np.sum(dw**2) + db**2
                
                for _ in range(10):
                    w_new = self.w - alpha * dw
                    b_new = self.b - alpha * db
                    new_loss = self._compute_loss(X, y_encoded, w_new, b_new, sample_weights)
                    
                    # Kiểm tra điều kiện giảm biên Armijo
                    if new_loss <= current_loss - c * alpha * grad_sq_norm:
                        break
                    alpha *= beta
                    
                # Cập nhật trọng số bằng learning rate tối ưu tìm được
                self.w -= alpha * dw
                self.b -= alpha * db
            
        return self
        
    def predict_proba(self, X):
        z = X.dot(self.w) + self.b
        prob_class_1 = self._sigmoid(z)
        prob_class_0 = 1.0 - prob_class_1
        
        # Trả về ma trận xác suất [P(y=0), P(y=1)] giống scikit-learn
        return np.column_stack((prob_class_0, prob_class_1))
        
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
