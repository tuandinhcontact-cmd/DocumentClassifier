from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data1" / "train_test_data"

file_path = DATA_DIR / "X_train_postSMOTE.pkl"

print("Current working directory:", Path.cwd())
print("BASE_DIR:", BASE_DIR)
print("DATA_DIR:", DATA_DIR)
print("FILE PATH:", file_path)
print("Exists:", file_path.exists())

if file_path.exists():
    print("Size:", file_path.stat().st_size, "bytes")

    with open(file_path, "rb") as f:
        first_bytes = f.read(100)

    print("First bytes:", first_bytes)
else:
    print("Không tìm thấy file.")