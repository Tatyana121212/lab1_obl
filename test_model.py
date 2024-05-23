import joblib
from sklearn.pipeline import Pipeline

def load_model(path):
    with open(path, 'rb') as file:
        return joblib.load(file)

def save_model(model, path):
    with open(path, 'wb') as file:
        joblib.dump(model, file)

# Укажите правильный путь к вашей модели
model_path = 'C:/oblaka/models/models/auto_mpg_model.pkl'

try:
    # Загрузите модель
    model = load_model(model_path)
    print("Модель успешно загружена.")
    
    # Сохраните модель заново
    save_model(model, model_path)
    print("Модель успешно пересохранена.")
except Exception as e:
    print(f"Ошибка при загрузке или сохранении модели: {e}")
