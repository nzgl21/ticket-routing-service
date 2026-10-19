import time
import json
import numpy as np
import onnxruntime as ort
from pathlib import Path
from transformers import AutoTokenizer

# 1. Пути к файлам
ONNX_MODEL_PATH = "/Users/arslanniazgulov/Documents/nlp_projects/ticket_routing_service/inference/distilbert_banking77/model_quantized.onnx"   # или "models/model.onnx"
TOKENIZER_DIR = "/Users/arslanniazgulov/Documents/nlp_projects/ticket_routing_service/models/distilbert-base-uncased_weighet_loss_lr3e-5_ep4"  # папка с токенизатором
CONFIG_PATH = Path(TOKENIZER_DIR) / "config.json"

# 2. Загружаем токенизатор и маппинг классов
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

with open(Path.cwd() / 'src/id2label.json', "r", encoding="utf-8") as f:
    id2label = json.load(f)

# 3. Создаём сессию ONNX Runtime на CPU
print(f"🔄 Загрузка ONNX модели: {ONNX_MODEL_PATH}...")
session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
print("✅ Модель готова к работе!\n")


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def predict_ticket(text: str):
    t0 = time.time()
    
    # Токенизация (сразу в формат numpy, без PyTorch!)
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="np"  # <-- numpy массивы!
    )
    
    # Передаём в ONNX
    onnx_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }
    
    # Инференс в C++ рантайме
    logits = session.run(None, onnx_inputs)[0]
    
    # Вероятности
    probs = softmax(logits)[0]
    pred_id = int(np.argmax(probs))
    confidence = float(probs[pred_id])
    category = id2label.get(str(pred_id), f"LABEL_{pred_id}")
    
    latency_ms = (time.time() - t0) * 1000
    
    return {
        "text": text,
        "category": category,
        "confidence": f"{confidence:.2%}",
        "latency_ms": f"{latency_ms:.2f} ms"
    }


# === Тестируем на реальных вопросах ===
test_queries = [
    "I lost my credit card yesterday, please block it!",
    "Why was I charged an extra fee on my international transfer?",
    "How can I change my PIN code in the app?",
    "When will my new card arrive in the mail?"
]

for query in test_queries:
    res = predict_ticket(query)
    print(f"Текст:       '{res['text']}'")
    print(f"Категория:   {res['category']}")
    print(f"Уверенность: {res['confidence']} | Скорость: {res['latency_ms']}")
    print("-" * 50)