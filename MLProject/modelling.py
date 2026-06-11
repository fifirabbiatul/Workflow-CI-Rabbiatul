import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def main():
    # Lokasi dataset saat dijalankan via MLflow project
    data_path = 'titanic_preprocessing.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset {data_path} tidak ditemukan!")

    X_train, X_test, y_train, y_test = load_data(data_path)

    # Mengaktifkan autolog untuk menyimpan model secara otomatis
    mlflow.autolog()

    with mlflow.start_run(run_name="CI_Retraining_Run") as run:
        print("Melatih model RandomForest...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Akurasi Model: {acc}")

        # Simpan Run ID ke sebuah file agar mudah dibaca oleh GitHub Actions (untuk proses build-docker)
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)
            
        print(f"Pelatihan selesai. Run ID: {run.info.run_id}")

if __name__ == "__main__":
    main()
