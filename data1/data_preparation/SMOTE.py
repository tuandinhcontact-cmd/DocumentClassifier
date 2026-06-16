from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

DATA_DIR = PROJECT_ROOT / "data1" / "train_test_data"

X_train = joblib.load(DATA_DIR / "X_train_preSMOTE.pkl")
y_train = joblib.load(DATA_DIR / "y_train_preSMOTE.pkl")

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

joblib.dump(X_train_smote, DATA_DIR / "X_train_postSMOTE.pkl")
joblib.dump(y_train_smote, DATA_DIR / "y_train_postSMOTE.pkl")

print("Saved SMOTE train data!")
print("Before SMOTE:", X_train.shape, y_train.shape)
print("After SMOTE:", X_train_smote.shape, y_train_smote.shape)