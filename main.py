import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from PIL import Image, ImageOps
import io

app = FastAPI()

# Загрузка обученной модели
model = tf.keras.models.load_model('fashion_model1.keras')

# Список классов Fashion-MNIST
classes = ['Футболка / топ', 'Брюки', 'Свитер', 'Платье', 'Пальто', 'Сандалии', 'Рубашка', 'Кроссовки', 'Сумка', 'Ботильоны']

def prepare_image(image_bytes, invert: bool):
    # Открываем и переводим в ЧБ
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    
    # 1. Инвертируем только если пользователь указал, что фото на светлом фоне
    if invert:
        img = ImageOps.invert(img)
    
    # 2. Автокадрирование: находим границы светлого объекта и обрезаем пустой черный фон
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
        
    # 3. Вписываем обрезанную вещь в квадрат 28x28 с сохранением пропорций, 
    # заполняя пустые места черным цветом (0)
    img = ImageOps.pad(img, (28, 28), color=0, method=Image.Resampling.LANCZOS)
    
    pixels = np.asarray(img, dtype=np.float32)
    
    # 4. Подавляем шум и слабые тени (все, что тусклее 50, делаем полностью черным)
    pixels[pixels < 50] = 0.0
    
    # Нормализация
    return (pixels / 255.0)[None, ..., None]

@app.get("/")
def read_index():
    # Отдаем HTML-страницу при обращении к корню сайта
    return FileResponse("index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...), invert: bool = Form(...)):
    image_bytes = await file.read()
    
    # Передаем параметр invert в функцию подготовки картинки
    processed_image = prepare_image(image_bytes, invert)
    
    # Предсказание
    probabilities = model.predict(processed_image)[0]
    predicted_class = int(np.argmax(probabilities))
    
    return {"class_name": classes[predicted_class], "confidence": float(probabilities[predicted_class])}