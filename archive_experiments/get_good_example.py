import pickle
import pandas as pd
from train_3step_cascade import ThreeStepCascadeClassifier, CustomOneVsRestClassifier, CustomMultiClassVotingClassifier

with open("cascade_model.pkl", "rb") as f:
    model_data = pickle.load(f)

clf = model_data['model']
vectorizer = model_data['vectorizer']

df = pd.read_csv("merged_cleaned_dataset.csv")
politics_docs = df[df['category'] == 'Politics and society'].head(50)

for idx, row in politics_docs.iterrows():
    text = row['cleaned_text']
    X_vec = vectorizer.transform([text])
    pred = clf.predict(X_vec)[0]
    if pred == 'Politics and society':
        print("=== GOOD EXAMPLE ===")
        print(f"Content: {row['cleaned_text']}")
        print("====================")
        break
