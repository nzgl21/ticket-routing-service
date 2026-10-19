from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from pathlib import Path

# 1. Загружаем модель сразу в ONNX
model = ORTModelForSequenceClassification.from_pretrained(
    "/Users/arslanniazgulov/Documents/nlp_projects/ticket_routing_service/models/distilbert-base-uncased_weighet_loss_lr3e-5_ep4", 
    export=True
)

# 2. Настраиваем квантование под процессор (AVX-512, VNNI или ARM)
quantizer = ORTQuantizer.from_pretrained(model)
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

# 3. Квантуем и сохраняем
quantizer.quantize(
    save_dir="inference/distilbert_banking77",
    quantization_config=qconfig
)