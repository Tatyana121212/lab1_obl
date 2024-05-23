import os
import logging
import pandas as pd
import numpy as np
from pickle import load
from sklearn.pipeline import Pipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError

class Instance(BaseModel):
    cylinders: int
    displacement: float
    horsepower: float
    weight: float
    acceleration: float
    model_year: int
    origin: int

app = FastAPI()

model_path: str = "C:\\oblaka\\models\\models\\auto_mpg_model.pkl"
if not model_path:
    raise ValueError("Переменная окружения $MODEL_PATH пуста!")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.get("/healthcheck")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/predictions")
async def predictions(instance: Instance) -> dict[str, float]:
    try:
        logging.info("Получены данные инстанса: %s", instance.dict())
        model = load_model(model_path)
        logging.info("Модель успешно загружена")
        instance_data = instance.dict()
        logging.info("Данные инстанса: %s", instance_data)
        prediction = make_inference(model, instance_data)
        logging.info("Успешно выполнено предсказание: %s", prediction)
        return prediction
    except ValidationError as ve:
        logging.error(f"Ошибка валидации данных: {ve}", exc_info=True)
        raise HTTPException(status_code=422, detail=ve.errors())
    except Exception as e:
        logging.error(f"Ошибка во время предсказания: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

def make_inference(in_model: Pipeline, in_data: dict) -> dict[str, float]:
    """Возвращает результат предсказания для in_data используя in_model."""
    try:
        logging.info("Входные данные для предсказания: %s", in_data)
        df = pd.DataFrame([in_data])
        logging.info("DataFrame для предсказания: %s", df)
        mpg = in_model.predict(df)
        logging.info("Результат предсказания (массив): %s", mpg)
        mpg_value = mpg[0] if isinstance(mpg, (np.ndarray, list)) else mpg
        logging.info("Извлеченное значение предсказания: %s", mpg_value)
        return {"mpg": round(float(mpg_value), 3)}
    except Exception as e:
        logging.error(f"Ошибка в make_inference: {e}", exc_info=True)
        raise

def load_model(path: str) -> Pipeline:
    """Возвращает модель, считанную из указанного пути."""
    try:
        logging.info("Загрузка модели из файла: %s", path)
        with open(path, "rb") as file:
            model: Pipeline = load(file)
        logging.info("Модель успешно загружена из файла")
        return model
    except Exception as e:
        logging.error(f"Ошибка при загрузке модели из {path}: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
