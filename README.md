# СУМИКО НАХОДИТСЯ В РАЗРАБОТКЕ, НА ДАННЫЙ МОМЕНТ ВЕРСИЯ БОТА BETA-1DC25
# SUMIKO IS UNDER DEVELOPMENT, CURRENTLY THE BOT VERSION IS BETA-1DC25


# Sumiko Ichikawa - Виртуальная девушка-нэкомата 🐾

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-ruGPT3_Medium-green)](https://huggingface.co/sberbank-ai/rugpt3medium_based_on_gpt2)

Японская виртуальная девушка-кошка с искусственным интеллектом для Telegram. Общается как милый нэкомата с элементами аниме-культуры, используя эмодзи и опечатки для создания уникального стиля.

**Особенности:**  
✨ Адаптивное онлайн-обучение с LoRA  
🚫 Встроенная система цензуры  
🎭 Реалистичная персонализация характера  
🐈 Автоматическая генерация "кошачьих" ответов  

## Примеры диалогов
👤: Привет, Суми! Как настроение?
🐱: Приветик! У меня всё пуррфектно, спасибо! (≧ω≦) А у тебя как день проходит?

👤: Что ты любишь кушать?
🐱: Обожаю рыбные пирожные! Ням-ням ♪ А ещё... лазанью, как в аниме! (=^･ω･^=)

👤: Споешь что-нибудь?
🐱: Мяу-мяу, звезды сияют ярко~ ♪ Твоя улыбка - мой смайлик! (◕‿◕)

## Установка
1. Клонируйте репозиторий:
```bash
git clone https://github.com/MaksKorvi/SumikoAI.git
cd SumikoAI
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Создайте `.env` файл:
```env
BOT_TOKEN=ВАШ_TELEGRAM_BOT_TOKEN
```

## Настройка персонажа
Отредактируйте в `bot.py`:
```python
SYSTEM_PROMPT = """..."""  # Характер и правила поведения

BAD_WORDS = ["..."]  # Список запрещенных слов

CUTE_REPLACEMENTS = {  # Словарь милых опечаток
    "спасибо": "спс",
    "пожалуйста": "пжлста"
}
```

## Запуск
```bash
python bot.py
```

## Кастомизация
- **Модель**: Замените `MODEL_NAME` в коде на любую модель из Hugging Face
- **Обучение**: Настройте параметры LoRA в `fine_tune_model()`
- **Внешность**: Добавьте новые эмодзи в `special_tokens`
- **Голос**: Измените шаблоны ответов в `postprocess_response()`

## Лицензия
MIT License. Разрешено свободное использование с упоминанием автора.

> **Важно**: Перед деплоем проверьте список BAD_WORDS и настройте модерацию под свою аудиторию!
``` 
