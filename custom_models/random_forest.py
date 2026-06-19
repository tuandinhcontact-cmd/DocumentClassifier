import numpy as np
from joblib import Parallel, delayed

class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature          # Chỉ số đặc trưng dùng để chia
        self.threshold = threshold      # Ngưỡng chia
        self.left = left                # Cây con bên trái
        self.right = right              # Cây con bên phải
        self.value = value              # Xác suất của lớp 1 (chỉ dùng cho nút lá)
        
    def is_leaf(self):
        return self.value is not None

class CustomDecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, max_features='sqrt'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None
        
    def _gini(self, y):
        # Tính toán Gini Impurity: Gini = 1 - sum(p_i^2)
        n = len(y)
        if n == 0:
            return 0.0
        p1 = np.sum(y == 1) / n
        p0 = 1.0 - p1
        return 1.0 - (p0**2 + p1**2)
        
    def _best_split(self, X, y, feat_idxs):
        best_gain = -1.0
        split_idx, split_thresh = None, None
        
        n_samples = X.shape[0]
        current_gini = self._gini(y)
        
        for feat in feat_idxs:
            # Lấy giá trị của đặc trưng thứ feat (X_column có thể là ma trận thưa)
            X_column = X[:, feat]
            if hasattr(X_column, "toarray"):
                X_column = X_column.toarray().ravel()
            else:
                X_column = np.array(X_column).ravel()
                
            # Trích xuất các giá trị khác không
            non_zero_vals = X_column[X_column > 0]
            if len(non_zero_vals) == 0:
                continue
                
            # Lấy ngẫu nhiên tối đa 5 ngưỡng từ các giá trị khác không
            unique_vals = np.unique(non_zero_vals)
            if len(unique_vals) > 5:
                thresholds = np.random.choice(unique_vals, 5, replace=False)
            else:
                thresholds = unique_vals
                
            # Thêm ngưỡng 0.0
            thresholds = np.append(thresholds, 0.0)
                
            for thresh in thresholds:
                left_mask = X_column <= thresh
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                    
                # Tính Gini sau khi chia
                gini_left = self._gini(y[left_mask])
                gini_right = self._gini(y[right_mask])
                
                n_left = np.sum(left_mask)
                n_right = n_samples - n_left
                weighted_gini = (n_left / n_samples) * gini_left + (n_right / n_samples) * gini_right
                
                # Tính lượng Gini giảm đi (Gini Gain)
                gain = current_gini - weighted_gini
                
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat
                    split_thresh = thresh
                    
        return split_idx, split_thresh
        
    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        
        # Điều kiện dừng đệ quy
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(np.unique(y)) == 1):
            leaf_value = np.sum(y == 1) / n_samples if n_samples > 0 else 0.0
            return DecisionNode(value=leaf_value)
            
        # Chọn ngẫu nhiên một số lượng đặc trưng con theo cấu hình max_features
        if self.max_features == 'sqrt':
            max_f = int(np.sqrt(n_features))
        else:
            max_f = n_features
        feat_idxs = np.random.choice(n_features, max_f, replace=False)
        
        # Tìm điểm chia tốt nhất
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        
        if best_feat is None:
            leaf_value = np.sum(y == 1) / n_samples
            return DecisionNode(value=leaf_value)
            
        # Chia dữ liệu và xây dựng cây con trái/phải
        X_col = X[:, best_feat]
        if hasattr(X_col, "toarray"):
            X_col = X_col.toarray().ravel()
        else:
            X_col = np.array(X_col).ravel()
            
        left_mask = X_col <= best_thresh
        right_mask = ~left_mask
        
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return DecisionNode(feature=best_feat, threshold=best_thresh, left=left_child, right=right_child)
        
    def fit(self, X, y):
        y_arr = np.array(y)
        self.root = self._build_tree(X, y_arr)
        return self
        
    def _predict_sample(self, node, x):
        if node.is_leaf():
            return node.value
        
        # Lấy giá trị đặc trưng của mẫu x (hỗ trợ cả ma trận thưa scipy)
        val = x[0, node.feature] if hasattr(x, "tocsr") else x[node.feature]
        
        if val <= node.threshold:
            return self._predict_sample(node.left, x)
        else:
            return self._predict_sample(node.right, x)
            
    def predict_proba(self, X):
        n_samples = X.shape[0]
        probs = np.zeros(n_samples)
        
        # Duyệt cây cho từng mẫu dữ liệu
        for idx in range(n_samples):
            # Lấy dòng thứ idx
            x = X[idx]
            probs[idx] = self._predict_sample(self.root, x)
            
        return np.column_stack((1.0 - probs, probs))

def _train_single_tree(X, y_arr, max_depth, min_samples_split, max_features, seed):
    # Thiết lập seed ngẫu nhiên cho tiến trình con
    np.random.seed(seed)
    n_samples = X.shape[0]
    bootstrap_idxs = np.random.choice(n_samples, n_samples, replace=True)
    X_bootstrap = X[bootstrap_idxs]
    y_bootstrap = y_arr[bootstrap_idxs]
    
    tree = CustomDecisionTree(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        max_features=max_features
    )
    tree.fit(X_bootstrap, y_bootstrap)
    return tree

class CustomRandomForest:
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2, max_features='sqrt', n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.trees = []
        
    def fit(self, X, y):
        self.trees = []
        y_arr = np.array(y)
        
        # Tạo danh sách các seed ngẫu nhiên cho mỗi cây con
        seeds = np.random.randint(0, 1000000, size=self.n_estimators)
        
        # Huấn luyện song song n_estimators cây quyết định bằng joblib
        self.trees = Parallel(n_jobs=self.n_jobs)(
            delayed(_train_single_tree)(
                X, y_arr, self.max_depth, self.min_samples_split, self.max_features, seeds[i]
            ) for i in range(self.n_estimators)
        )
            
        return self
        
    def predict_proba(self, X):
        # Tính toán xác suất dự đoán trung bình của tất cả các cây trong rừng
        all_tree_probs = []
        for tree in self.trees:
            all_tree_probs.append(tree.predict_proba(X))
            
        # Lấy trung bình cộng
        mean_probs = np.mean(all_tree_probs, axis=0)
        return mean_probs
        
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
