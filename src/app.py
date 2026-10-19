import os
import json
import time
import numpy as np
import onnxruntime as ort
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

# 1. Пути к моделям
ONNX_MODEL_PATH = "inference/distilbert_banking77/model_quantized.onnx"   # или "models/model.onnx"
TOKENIZER_DIR = "models/distilbert-base-uncased_weighet_loss_lr3e-5_ep4"
CONFIG_PATH = os.path.join("src/id2label.json")


session = None
tokenizer = None
id2label = None


# 2. Жизненный цикл сервиса (Lifespan): загрузка модели ПРИ СТАРТЕ
@asynccontextmanager
async def lifespan(app: FastAPI):
    global session, tokenizer, id2label
    print("🔄 Загрузка токенизатора и ONNX модели...")
    
    # Загружаем токенизатор
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    
    # Загружаем маппинг классов из config.json
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        id2label = json.load(f)

        
    # Загружаем быстрый C++ движок ONNX
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
    
    print(f"🚀 Модель успешно загружена! Всего классов: {len(id2label)}")
    yield
    print("🛑 Остановка сервиса...")


# 3. Инициализируем FastAPI с метаданными
app = FastAPI(
    title="Customer Ticket Routing API",
    description="Классификация клиентских обращений в банк с задержкой < 10 мс",
    version="1.0.0",
    lifespan=lifespan
)


# 4. Схемы валидации данных (Pydantic)
# Пользователь обязан прислать не пустую строку
class TicketRequest(BaseModel):
    text: str = Field(..., example="I lost my credit card yesterday, please block it!")

# Формат ответа клиенту
class TicketResponse(BaseModel):
    category: str
    confidence: float
    latency_ms: float


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


# 5. Эндпоинты API

@app.get("/health")
def healthcheck():
    """Проверка доступности сервиса для Docker / Kubernetes"""
    return {"status": "ok", "classes_count": len(id2label) if id2label else 0}


@app.post("/predict", response_model=TicketResponse)
async def predict_ticket(request: TicketRequest):
    """Классификация одного обращения"""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Текст обращения не может быть пустым.")
        
    t0 = time.time()
    
    # Токенизация (сразу в numpy)
    inputs = tokenizer(
        request.text, 
        padding="max_length", 
        truncation=True, 
        max_length=64, 
        return_tensors="np"
    )
    
    onnx_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }
    
    # Инференс в ONNX Runtime
    logits = session.run(None, onnx_inputs)[0]
    
    # Вычисление вероятностей
    probs = softmax(logits)[0]
    pred_id = int(np.argmax(probs))
    confidence = float(probs[pred_id])
    category = id2label.get(pred_id, f"LABEL_{pred_id}")
    
    latency = (time.time() - t0) * 1000  # в миллисекундах
    
    return TicketResponse(
        category=category,
        confidence=round(confidence, 4),
        latency_ms=round(latency, 2)
    )