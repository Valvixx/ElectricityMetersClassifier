# SECON Meter OCR

Проект для распознавания показаний электросчётчиков по изображению. Репозиторий объединяет API на Flask, модели классификации, сегментации и OCR, а также локальные скрипты для обучения и оценки качества.

Основная идея пайплайна:

1. Изображение счётчика попадает в API.
2. Классификатор определяет тип счётчика: `old` или `new`.
3. Для выбранного типа используется своя YOLO-модель сегментации, которая вырезает область с показанием.
4. Для старых счётчиков OCR идёт через комбинированный пайплайн:
   `printed`/`handwritten` сначала определяет отдельный классификатор, затем выбирается соответствующая TrOCR-модель.
5. Для новых счётчиков используется отдельная TrOCR-модель `new-printed`.

## Технологии

- Python 3
- Flask + Flasgger для REST API и Swagger UI
- PyTorch и Torchvision для классификаторов
- Hugging Face Transformers для TrOCR
- Ultralytics YOLOv8 для детекции/сегментации области счётчика

Ключевые зависимости зафиксированы в [requirements.txt](requirements.txt).

## Структура проекта

- [main.py](main.py) - Flask API и production-пайплайн инференса
- [classification.py](classification.py) - обучение и загрузка классификаторов на базе ResNet18
- [support_scripts/train_trocr.py](support_scripts/train_trocr.py) - обучение OCR-моделей TrOCR
- [support_scripts/train_ocr_type_classifier.py](support_scripts/train_ocr_type_classifier.py) - обучение классификатора `printed` / `handwritten`
- [support_scripts/evaluate_ocr_models.py](support_scripts/evaluate_ocr_models.py) - оценка OCR и экспорт результатов в CSV
- [support_scripts/ocr_eval_results/combined_local3.csv](support_scripts/ocr_eval_results/combined_local3.csv) - основной подробный отчёт по качеству

## Как работает сервис

Точка входа API: `POST /batch-process`.

На каждом изображении выполняется такой маршрут:

1. `meter_classifier.pth` определяет, старый это счётчик или новый.
2. Если счётчик старый, выбирается YOLO-модель из `runs/segment/train3/weights/best.pt`.
3. Если счётчик новый, выбирается YOLO-модель из `runs/segment/train5/weights/best.pt`.
4. После кропа:
   - `old` -> `combined_local`
   - `new` -> `new_printed_local`
5. Для `combined_local`:
   - `ocr_type_classifier3.pth` определяет `printed` или `handwritten`
   - затем запускается одна из двух TrOCR-моделей:
     - `support_scripts/ocr_checkpoints/printed/best`
     - `support_scripts/ocr_checkpoints/handwritten/best`

Дополнительно в генерации OCR включены ограничения на допустимые токены:

- разрешаются только цифровые токены и специальные служебные токены
- выход ограничен диапазоном от 5 до 10 цифр
- используется beam search с `num_beams=5`
- среди кандидатов выбирается наиболее правдоподобная цифровая последовательность

## Архитектура моделей

### 1. Классификаторы типа изображения

И для классификатора типа счётчика, и для классификатора OCR-типа используется `ResNet18` с заменой последнего полносвязного слоя на задачу из 2 классов.

Структура модели:

- `conv1`: `Conv2d(3, 64, kernel_size=7x7, stride=2, padding=3)`
- `bn1`: `BatchNorm2d(64)`
- `relu`
- `maxpool`: `3x3`, `stride=2`
- `layer1`: 2 residual `BasicBlock` на 64 каналах
- `layer2`: 2 residual `BasicBlock`, переход `64 -> 128`
- `layer3`: 2 residual `BasicBlock`, переход `128 -> 256`
- `layer4`: 2 residual `BasicBlock`, переход `256 -> 512`
- `avgpool`: `AdaptiveAvgPool2d(1x1)`
- `fc`: `Linear(512 -> 2)`

Перед подачей в сеть изображение:

- приводится к размеру `224x224`
- переводится в tensor
- нормализуется по ImageNet-статистикам

Во время обучения используются аугментации:

- небольшой поворот
- случайный crop/pad
- изменение контраста, яркости и резкости
- Gaussian blur и Median filter
- salt-and-pepper noise
- autocontrast

### 2. OCR-модели

OCR построен на `VisionEncoderDecoderModel` из Transformers. Все локальные чекпоинты в репозитории используют одну и ту же базовую архитектуру TrOCR:

- энкодер: ViT
- декодер: TrOCR decoder
- общий размер модели: `333,921,792` параметров

Структура OCR-модели по сохранённым конфигам:

- `encoder.model_type`: `vit`
- `image_size`: `384`
- `patch_size`: `16`
- `num_channels`: `3`
- `num_hidden_layers`: `12`
- `hidden_size`: `768`
- `num_attention_heads`: `12`
- `intermediate_size`: `3072`

- `decoder.model_type`: `trocr`
- `decoder_layers`: `12`
- `d_model`: `1024`
- `decoder_attention_heads`: `16`
- `decoder_ffn_dim`: `4096`
- `max_position_embeddings`: `512`
- `vocab_size`: `50265`

Используемые OCR-модели:

- `printed` для печатных старых счётчиков
- `handwritten` для рукописных старых счётчиков
- `new-printed` для новых счётчиков

### 3. Сегментация области счётчика

Для вырезания области показаний используются две модели Ultralytics YOLOv8:

- одна для `old`
- одна для `new`

На этапе инференса берётся первый найденный bounding box и по нему выполняется crop. Это позволяет отделить задачу локализации области считывания от самой OCR-модели.

## Обучение

### OCR

Скрипт [support_scripts/train_trocr.py](support_scripts/train_trocr.py) обучает TrOCR с такими ключевыми параметрами:

- базовая модель: `microsoft/trocr-base-printed`
- `epochs=14`
- `batch_size=1`
- `grad_accum_steps=4`
- `learning_rate=1e-5`
- `max_target_length=16`
- `num_beams=5`
- `early_stopping_patience=4`
- включены аугментации
- включён `gradient checkpointing`
- включён AMP при наличии CUDA

Оптимизатор: `Adafactor`.

Основные метрики в ходе валидации:

- `val_loss`
- `CER` (Character Error Rate)
- `exact match`

### Классификатор OCR-типа

Скрипт [support_scripts/train_ocr_type_classifier.py](support_scripts/train_ocr_type_classifier.py) обучает бинарный классификатор `printed` / `handwritten`:

- `epochs=15`
- `batch_size=32`
- `learning_rate=1e-4`
- аугментации включены

### Классификатор типа счётчика

Скрипт [classification.py](classification.py) также обучает ResNet18 для классов `new` / `old`.

В коде сохраняется лучший checkpoint по `val accuracy`.

## Результаты

Ниже приведены результаты, которые подтверждаются файлом [support_scripts/ocr_eval_results/combined_local3.csv](support_scripts/ocr_eval_results/combined_local3.csv). Это самая надёжная таблица в репозитории для описания достигнутого качества.

### Общие результаты по движкам

| Движок | Кол-во образцов | Exact match | Avg edit distance | Avg CER |
|---|---:|---:|---:|---:|
| `combined_local` | 45 | 62.22% | 0.822 | 0.1139 |
| `new_printed_local` | 19 | 89.47% | 0.105 | 0.0132 |

Вывод: лучший зафиксированный результат в текущем репозитории показывает `new_printed_local` на выборке новых печатных счётчиков: `89.47%` точных совпадений.

### Результаты внутри `combined_local`

| Поднабор | Кол-во образцов | Exact match | Avg edit distance | Avg CER |
|---|---:|---:|---:|---:|
| Старые печатные (`printed`) | 25 | 60.00% | 0.920 | 0.1305 |
| Старые рукописные (`handwritten`) | 20 | 65.00% | 0.700 | 0.0932 |

### Качество маршрутизации `printed` / `handwritten`

Для смешанного движка `combined_local` можно отдельно оценить работу классификатора OCR-типа:

| Ожидаемый тип | Предсказанный маршрут | Кол-во |
|---|---:|---:|
| `printed` | `printed` | 20 |
| `printed` | `handwritten` | 5 |
| `handwritten` | `handwritten` | 17 |
| `handwritten` | `printed` | 3 |

Итоговая точность маршрутизации по этой выборке: `37 / 45 = 82.22%`.

### Что означают метрики

- `Exact match` - доля изображений, где показание распознано полностью без ошибок.
- `Edit distance` - среднее количество символьных правок до правильного ответа.
- `CER` - относительная символьная ошибка; чем меньше, тем лучше.

## Интерпретация результатов

На текущем этапе проект уже показывает рабочее качество, особенно на новых печатных счётчиках. Это означает, что связка:

- классификация типа счётчика
- YOLO-crop
- специализированная TrOCR-модель

работает лучше всего там, где структура цифр стабильна и близка к печатному шаблону.

Для старых счётчиков качество ниже, но остаётся практическим:

- рукописные показания распознаются немного лучше печатных в mixed-оценке
- часть ошибок приходит не только из OCR, но и из неправильной маршрутизации между `printed` и `handwritten`
- самые тяжёлые ошибки обычно выглядят как потеря первой цифры, вставка лишней цифры или сдвиг последовательности

## Пример запроса

```bash
curl -X POST http://localhost:5001/batch-process \
  -F "images=@Images/80010_1710342959282.jpg" \
  -F "images=@Images/83290_1712397619169.jpg"
```

Пример ответа:

```json
[
  {
    "image": "80010_1710342959282.jpg",
    "meter_type": "old",
    "prediction": "1234567"
  }
]
```
