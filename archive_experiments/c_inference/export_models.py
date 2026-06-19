import os
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def export_to_c(X_sample, scaler, lr, gnb, svm, rf, filename="model_parameters.h"):
    num_features = X_sample.shape[1]
    
    with open(filename, "w") as f:
        f.write("/* Model parameters generated automatically by export_models.py */\n")
        f.write("#ifndef MODEL_PARAMETERS_H\n")
        f.write("#define MODEL_PARAMETERS_H\n\n")
        
        f.write(f"#define NUM_FEATURES {num_features}\n\n")
        
        # 1. StandardScaler Parameters
        f.write("/* StandardScaler Parameters */\n")
        f.write(f"const double SCALER_MEAN[NUM_FEATURES] = {{ {', '.join(map(str, scaler.mean_))} }};\n")
        f.write(f"const double SCALER_SCALE[NUM_FEATURES] = {{ {', '.join(map(str, scaler.scale_))} }};\n\n")
        
        # 2. Logistic Regression
        f.write("/* Logistic Regression Parameters */\n")
        f.write(f"const double LR_COEF[NUM_FEATURES] = {{ {', '.join(map(str, lr.coef_[0]))} }};\n")
        f.write(f"const double LR_INTERCEPT = {lr.intercept_[0]};\n\n")
        
        # 3. Gaussian Naive Bayes
        f.write("/* Gaussian Naive Bayes Parameters */\n")
        f.write(f"const double GNB_CLASS_PRIOR[2] = {{ {gnb.class_prior_[0]}, {gnb.class_prior_[1]} }};\n")
        f.write(f"const double GNB_THETA[2][NUM_FEATURES] = {{\n")
        f.write(f"    {{ {', '.join(map(str, gnb.theta_[0]))} }},\n")
        f.write(f"    {{ {', '.join(map(str, gnb.theta_[1]))} }}\n}};\n")
        f.write(f"const double GNB_VAR[2][NUM_FEATURES] = {{\n")
        f.write(f"    {{ {', '.join(map(str, gnb.var_[0]))} }},\n")
        f.write(f"    {{ {', '.join(map(str, gnb.var_[1]))} }}\n}};\n\n")
        
        # 4. Support Vector Machine (Linear SVM with Platt Scaling calibration)
        # We export the linear decision function boundary (w^T x + b) 
        # and Platt scaling parameters (A, B) to output probability: P = 1 / (1 + exp(A*f(x) + B))
        f.write("/* Support Vector Machine (Linear Kernel) Parameters */\n")
        # For linear SVM, w is computed as dual_coef_ @ support_vectors_
        w = svm.dual_coef_ @ svm.support_vectors_
        b = svm.intercept_[0]
        f.write(f"const double SVM_COEF[NUM_FEATURES] = {{ {', '.join(map(str, w[0]))} }};\n")
        f.write(f"const double SVM_INTERCEPT = {b};\n")
        # Platt scaling coefficients (A, B) for probability prediction
        # svm.probA_ and svm.probB_ are generated when probability=True
        f.write(f"const double SVM_PROB_A = {svm.probA_[0]};\n")
        f.write(f"const double SVM_PROB_B = {svm.probB_[0]};\n\n")
        
        # 5. Random Forest Parameters
        f.write("/* Random Forest Parameters */\n")
        f.write(f"#define RF_NUM_TREES {rf.n_estimators}\n\n")
        
        # Write structures for tree nodes
        f.write("typedef struct {\n")
        f.write("    int feature;      /* Feature index to split on (-1 for leaf) */\n")
        f.write("    double threshold; /* Split threshold */\n")
        f.write("    int left;         /* Index of left child node */\n")
        f.write("    int right;        /* Index of right child node */\n")
        f.write("    double value;     /* Class 1 probability value for leaf node */\n")
        f.write("} Node;\n\n")
        
        # Export all trees
        for tree_idx, estimator in enumerate(rf.estimators_):
            tree = estimator.tree_
            num_nodes = tree.node_count
            f.write(f"/* Tree {tree_idx} Nodes count: {num_nodes} */\n")
            f.write(f"const Node RF_TREE_{tree_idx}[{num_nodes}] = {{\n")
            for node_idx in range(num_nodes):
                feature = tree.feature[node_idx]
                threshold = tree.threshold[node_idx]
                left = tree.children_left[node_idx]
                right = tree.children_right[node_idx]
                
                # If leaf node, compute class probability
                if left == -1 and right == -1:
                    value_arr = tree.value[node_idx][0]
                    prob = value_arr[1] / np.sum(value_arr) # Probability of class 1
                else:
                    prob = 0.0
                
                f.write(f"    {{ {feature}, {threshold}, {left}, {right}, {prob} }}")
                if node_idx < num_nodes - 1:
                    f.write(",\n")
                else:
                    f.write("\n")
            f.write("};\n\n")
            
        # Write array of trees pointers and their sizes
        f.write("const Node* const RF_TREES[RF_NUM_TREES] = {\n")
        for tree_idx in range(rf.n_estimators):
            f.write(f"    RF_TREE_{tree_idx}")
            if tree_idx < rf.n_estimators - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("};\n\n")
        
        f.write("#endif\n")
    print(f"Đã xuất thành công tham số mô hình vào file C Header: {filename}")

def main():
    print("=== Đang huấn luyện mô hình trong Python để xuất ra C ===")
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Fit Models
    lr = LogisticRegression(random_state=42).fit(X_train_scaled, y_train)
    gnb = GaussianNB().fit(X_train_scaled, y_train)
    # SVM Linear kernel with probability
    svm = SVC(kernel='linear', probability=True, random_state=42).fit(X_train_scaled, y_train)
    # Random Forest with 5 trees (to keep header file size small and clean)
    rf = RandomForestClassifier(n_estimators=5, random_state=42).fit(X_train_scaled, y_train)
    
    # Export parameters
    export_to_c(X, scaler, lr, gnb, svm, rf, filename="model_parameters.h")
    
    # Print a test sample for C verification
    sample_idx = 0
    test_sample = X_test[sample_idx]
    scaled_sample = scaler.transform([test_sample])[0]
    print(f"\nMẫu dữ liệu kiểm thử (Chỉ số {sample_idx}, Nhãn thực tế: {y_test[sample_idx]}):")
    print("Mảng giá trị C thuần để copy/paste vào chương trình C:")
    print(f"double test_features[NUM_FEATURES] = {{ {', '.join(map(str, test_sample))} }};")
    
    print("\nXác suất dự đoán lớp 1 (Benign) của scikit-learn:")
    print(f"Logistic Regression: {lr.predict_proba([scaled_sample])[0][1]:.6f}")
    print(f"Gaussian NB: {gnb.predict_proba([scaled_sample])[0][1]:.6f}")
    print(f"SVM: {svm.predict_proba([scaled_sample])[0][1]:.6f}")
    print(f"Random Forest: {rf.predict_proba([scaled_sample])[0][1]:.6f}")
    print(f"Soft Voting Ensemble: {(lr.predict_proba([scaled_sample])[0][1] + gnb.predict_proba([scaled_sample])[0][1] + svm.predict_proba([scaled_sample])[0][1] + rf.predict_proba([scaled_sample])[0][1])/4:.6f}")

if __name__ == "__main__":
    main()
