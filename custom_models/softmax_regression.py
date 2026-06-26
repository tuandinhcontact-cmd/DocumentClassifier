import numpy as np

class CustomSoftmaxRegression:
    def __init__(self, lr=0.01, epochs=100, C=1.0, class_weight='balanced', batch_size=1024, beta1=0.9, beta2=0.999, eps=1e-8, verbose=False):
        self.lr = lr
        self.epochs = epochs
        self.C = C  # Tham số điều chuẩn L2 (Càng nhỏ thì L2 càng lớn)
        self.class_weight = class_weight
        self.batch_size = batch_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.verbose = verbose
        self.w = None  # Ma trận trọng số (n_features x n_classes)
        self.b = None  # Vector bias (n_classes,)
        self.classes = None
        self.class_to_idx = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        n_samples, n_features = X.shape
        
        # Khởi tạo trọng số W (n_features x n_classes) và bias b (n_classes,)
        self.w = np.zeros((n_features, n_classes))
        self.b = np.zeros(n_classes)
        
        # One-hot encoding cho nhãn y
        y_idx = np.array([self.class_to_idx[val] for val in y])
        Y = np.zeros((n_samples, n_classes))
        Y[np.arange(n_samples), y_idx] = 1.0
        
        # Tính trọng số mẫu (sample weights) nếu class_weight='balanced'
        if self.class_weight == 'balanced':
            classes_in_y, counts = np.unique(y_idx, return_counts=True)
            class_weights = n_samples / (n_classes * counts)
            weight_dict = dict(zip(classes_in_y, class_weights))
            sample_weights = np.array([weight_dict[val] for val in y_idx]).reshape(-1, 1)
        else:
            sample_weights = np.ones((n_samples, 1))

        # Trộn ngẫu nhiên dữ liệu một lần trước khi huấn luyện (để slice nhanh hơn trong vòng lặp batch)
        shuffle_idx = np.random.permutation(n_samples)
        X_shuffled = X[shuffle_idx]
        Y_shuffled = Y[shuffle_idx]
        sw_shuffled = sample_weights[shuffle_idx]

        # Xác định kích thước lô (batch size)
        batch_size = self.batch_size if (self.batch_size is not None and self.batch_size < n_samples) else n_samples
        n_batches = int(np.ceil(n_samples / batch_size))

        # Khởi tạo các mảng mô-men cho thuật toán tối ưu hóa Adam
        m_w = np.zeros((n_features, n_classes))
        v_w = np.zeros((n_features, n_classes))
        m_b = np.zeros(n_classes)
        v_b = np.zeros(n_classes)
        
        t = 0
        # Vòng lặp huấn luyện tối ưu hóa tham số Adam theo từng lô (Mini-batch Adam)
        for epoch in range(1, self.epochs + 1):
            # Trộn thứ tự các lô ở mỗi epoch để tăng tính ngẫu nhiên
            batch_order = np.arange(n_batches)
            np.random.shuffle(batch_order)
            
            for b_idx in batch_order:
                start = b_idx * batch_size
                end = min(start + batch_size, n_samples)
                
                X_batch = X_shuffled[start:end]
                Y_batch = Y_shuffled[start:end]
                sw_batch = sw_shuffled[start:end]
                
                b_samples = end - start
                if b_samples <= 0:
                    continue
                
                # Tính toán logits: Z = X * W + b
                z = X_batch.dot(self.w) + self.b
                
                # Tính toán xác suất bằng hàm Softmax (áp dụng Max subtraction để tránh tràn số thực)
                z_shifted = z - np.max(z, axis=1, keepdims=True)
                exp_z = np.exp(z_shifted)
                probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
                
                # Tính độ lệch giữa xác suất dự đoán và nhãn one-hot (có nhân trọng số mẫu, chia cho kích thước lô hiện tại)
                E = (probs - Y_batch) * sw_batch / b_samples
                
                # Đạo hàm hàm lỗi (Gradient) đối với W và b
                # dw có kích thước (n_features x n_classes), quy chuẩn phạt L2 được chia cho n_samples toàn cục
                dw = X_batch.T.dot(E) + (1.0 / (self.C * n_samples)) * self.w
                # db có kích thước (n_classes,)
                db = np.sum(E, axis=0)
                
                # Tăng biến đếm bước t của Adam
                t += 1
                
                # Cập nhật mô-men cấp 1 và cấp 2 (Adam moving averages)
                m_w = self.beta1 * m_w + (1.0 - self.beta1) * dw
                v_w = self.beta2 * v_w + (1.0 - self.beta2) * (dw ** 2)
                m_b = self.beta1 * m_b + (1.0 - self.beta1) * db
                v_b = self.beta2 * v_b + (1.0 - self.beta2) * (db ** 2)
                
                # Hiệu chuẩn bias correction cho Adam
                m_w_hat = m_w / (1.0 - self.beta1 ** t)
                v_w_hat = v_w / (1.0 - self.beta2 ** t)
                m_b_hat = m_b / (1.0 - self.beta1 ** t)
                v_b_hat = v_b / (1.0 - self.beta2 ** t)
                
                # Cập nhật trọng số và bias
                self.w -= (self.lr / (np.sqrt(v_w_hat) + self.eps)) * m_w_hat
                self.b -= (self.lr / (np.sqrt(v_b_hat) + self.eps)) * m_b_hat
                
            # Tính và in hàm lỗi Categorical Cross-Entropy nếu verbose = True (tính trên tập dữ liệu toàn cục)
            if self.verbose and (epoch == 1 or epoch % 10 == 0 or epoch == self.epochs):
                z_full = X_shuffled.dot(self.w) + self.b
                z_full_shifted = z_full - np.max(z_full, axis=1, keepdims=True)
                exp_z_full = np.exp(z_full_shifted)
                probs_full = exp_z_full / np.sum(exp_z_full, axis=1, keepdims=True)
                
                loss_ce = -np.sum(Y_shuffled * np.log(probs_full + 1e-15)) / n_samples
                loss_l2 = 0.5 * (1.0 / (self.C * n_samples)) * np.sum(self.w ** 2)
                loss_total = loss_ce + loss_l2
                print(f"      Epoch {epoch:3d}/{self.epochs} - Loss CE: {loss_ce:.6f} - Total Loss (with L2): {loss_total:.6f}")
        return self


    def predict_proba(self, X):
        z = X.dot(self.w) + self.b
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]
