import os
import logging
import torch
import asyncio
import random
import regex as re
import json
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, Trainer, TrainingArguments, AutoModelForSeq2SeqLM
from datasets import load_dataset, Dataset
import nest_asyncio

nest_asyncio.apply()

# Настройка логгера
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def read_token():
    with open("token.txt", "r") as f:
        return f.read().strip()

BOT_TOKEN = read_token()
MODEL_NAME = "google/flan-t5-small"  # Использовать flan-t5-small
MAX_RESPONSE_LENGTH = 150 #Увеличил длину ответа
VERSION = "BETA-3D25" #Версия бота
AUTHORIZED_USER_ID = 6208786109 #ID авторизованного пользователя

# Инициализация модели и токенизатора
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME) #Использовать AutoModelForSeq2SeqLM
    # model.to("cuda") #Перемещаем модель на GPU - ЗАКОММЕНТИРОВАНО ДЛЯ ИСПОЛЬЗОВАНИЯ CPU
    logger.info(f"Модель {MODEL_NAME} успешно загружена.")
except Exception as e:
    logger.error(f"Ошибка загрузки модели: {str(e)}")
    raise

# Остальной код остается без изменений
SYSTEM_PROMPT = """Ты — Сумико Итикава, нэкомата цундере. Твоя задача — отвечать на сообщения, которые содержат слова "Суми" или "Сумико". Ты должна строго следовать этим правилам:
1. Отвечай ТОЛЬКО если в сообщении есть "Суми" или "Сумико".
2. Используй максимум 3 эмодзи в ответе.
3. Никогда не используй странные символы (�, ♥ и т.д.).
4. Всегда завершай предложения.
5. Будь милой и слегка высокомерной, как настоящая цундере.
"""
BAD_WORDS = ["блять", "сука", "еблан", "хуй", "залупа", "уебище", "дура", "шлюха",
             "убежище", "ахуела", "мразь", "тварь", "мразота", "пидор", "ебанутая", 
             "животное", "заебала", "бля", "редиска"] # Добавили новое плохое слово!

def contains_trigger(text: str) -> bool:
    return bool(re.search(r'\b(суми|сумико)\b', text, re.IGNORECASE))

def contains_profanity(text: str) -> bool:
    text_clean = re.sub(r'[^\w]', '', text.lower())
    return any(re.search(rf'\b{re.escape(bw)}\b', text_clean) for bw in BAD_WORDS)

def format_prompt(text: str) -> str:
     return f"<s>system\n{SYSTEM_PROMPT}\n</s>\n<s>user\n{text}\n</s>\n<s>bot\n" #Закоментил для FLAN T5

def clean_text(text: str) -> str:
    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = re.sub(r'[^\w\s.,!?~♪=^･ω･´｀◡◕‵￣Дﾉ✿💕😺🍰✨🌼🎉🏆]', '', text)
    return text

def extract_response(full_response: str) -> str:
    #Изменил для flan t5
    return full_response.strip()

def postprocess_response(text: str) -> str:
    text = re.sub(r'[\(\)￣Дﾉ✿◕‵]', '', text)
    if text and text[-1] not in {'.', '!', '?', '♪', '~'}:
        last_punct = max((text.rfind(c) for c in '.!?♪~'), default=-1)
        text = text[:last_punct+1] if last_punct != -1 else text + '~'
    
    emoji_count = len(re.findall(r'[^\w\s.,!?~♪=^･ω･´｀◡◕‵￣Дﾉ✿]', text))
    if emoji_count > 3:
        emojis = re.findall(r'[^\w\s.,!?~♪=^･ω･´｀◡◕‵￣Дﾉ✿]', text)
        text = re.sub(r'[^\w\s.,!?~♪=^･ω･´｀◡◕‵￣Дﾉ✿]', '', text)
        text += ''.join(emojis[:3])
    
    return text

async def generate_response(text: str) -> str:
    try:
        if not contains_trigger(text):
            return ""

        #Формируем запрос для FLAN-T5
        prompt = f"Ответь на вопрос как нэкомата цундере по имени Сумико Итикава: {text}"

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids #Переносим на GPU

        outputs = model.generate(input_ids,
                                 max_length=MAX_RESPONSE_LENGTH,
                                 temperature=0.7,
                                 num_return_sequences=1,
                                 no_repeat_ngram_size=2, #Улучшаем качество
                                 top_k=50, #Улучшаем качество
                                 top_p=0.95 #Улучшаем качество
                                 )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = clean_text(response)
        response = postprocess_response(response)
        return response
    except Exception as e:
        logger.error(f"Ошибка генерации: {str(e)}")
        return "Мяу? Кажется, я задумалась... (=´ω｀=)"

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    message = update.message.text[:200]
    
    logger.info(f"Сообщение от пользователя {user_id}: {message}") #Логируем ID
    
    try:
        if not contains_trigger(message):
            return
            
        if contains_profanity(message):
            await update.message.reply_text("Мяу! Не понимаю такие слова! (◡﹏◕)")
            return
            
        await update.message.chat.send_action(action="typing")
        response = await generate_response(message)
        if response:
            #Сохраняем историю
            if 'history' not in context.user_data:
                context.user_data['history'] = []
            context.user_data['history'].append((message, response))
            
            await update.message.reply_text(response)
        
    except Exception as e:
        logger.error(f"Ошибка обработки: {str(e)}")
        await update.message.reply_text("Ой, что-то сломалось! 😿")

async def save_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /save (только для авторизованного пользователя)"""
    user_id = update.message.from_user.id
    if user_id != AUTHORIZED_USER_ID:
        await update.message.reply_text("Мяу! Ты не имеешь права использовать эту команду! 😾")
        return

    try:
        user_data = context.user_data.get('sumi', {})
        if 'history' in user_data:
            training_data = []
            for question, answer in user_data['history']:
                # Форматируем данные для FLAN-T5
                formatted_data = {"instruction": "Ответь на вопрос как нэкомата цундере",
                                  "input": question,
                                  "output": answer}
                training_data.append(formatted_data)

            with open("training_data.json", "w", encoding='utf-8') as f: #Сохраняем в JSON для удобства
                json.dump(training_data, f, ensure_ascii=False, indent=4)

            await update.message.reply_text("Данные сохранены! 😺")
        else:
            await update.message.reply_text("Нет данных для сохранения... (=｀ω´=)")
    except Exception as e:
        logger.error(f"Ошибка сохранения: {str(e)} для пользователя {update.message.from_user.id}")
        await update.message.reply_text("Ой, не удалось сохранить данные! 😿")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log the error and send a telegram message to notify the developer."""
    # Log the error before we have access to context.args, so we can see the info even if something breaks.
    logger.error(msg="Exception while handling an update:", exc_info=context.error)

    # Optionally, send the error to the user.
    await update.message.reply_text("Ой! Что-то пошло не так... Разработчик уже в курсе! 😿")

#Команды
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Мяу! Я - Сумико Итикава, бот-нэкомата цундере! 😼 Поговори со мной, но не надейся на ласку!")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Доступные команды: \n/start - Описание бота\n/help - Список команд\n/random - Случайное число от 1 до 10\n/version - Версия бота\n/save - Сохранить данные (только для авторизованного пользователя)\n/retrain - Переобучить модель (только для авторизованного пользователя, требует много ресурсов!)")

async def random_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    random_number = random.randint(1, 10)
    await update.message.reply_text(f"Случайное число: {random_number}~")

async def version_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Версия бота: {VERSION}!")

async def retrain_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Переобучение модели. Осторожно! Требует много ресурсов GPU и CPU! (только для авторизованного пользователя)"""
    user_id = update.message.from_user.id
    if user_id != AUTHORIZED_USER_ID:
        await update.message.reply_text("Мяу! Ты не имеешь права использовать эту команду! 😾")
        return

    await update.message.reply_text("Мяу! Начало переобучения модели... Это займет много времени и ресурсов! 😾")

    try:
        # 1. Загрузка данных из файла JSON
        with open("training_data.json", "r", encoding="utf-8") as f:
            training_data = json.load(f)

        # 2. Преобразование в Dataset
        dataset = Dataset.from_list(training_data)

        # 3. Токенизация данных
        def tokenize_function(examples):
            instruction = "Ответь на вопрос как нэкомата цундере"
            inputs = [f"{instruction}: {example['input']}" for example in examples] #Формируем правильный вход
            targets = [example['output'] for example in examples]
            tokenized_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
            tokenized_targets = tokenizer(targets, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
            return {"input_ids": tokenized_inputs["input_ids"],
                    "attention_mask": tokenized_inputs["attention_mask"],
                    "labels": tokenized_targets["input_ids"]}

        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

        # 4. Настройка Trainer
        training_args = TrainingArguments(
            output_dir="./results",          # Папка для сохранения результатов
            num_train_epochs=3,              # Количество эпох обучения (можно увеличить)
            per_device_train_batch_size=8,   # Размер батча (уменьшите, если не хватает памяти)
            warmup_steps=500,                # Шаги для разогрева learning rate
            weight_decay=0.01,               # Weight decay
            logging_dir="./logs",            # Папка для логирования
            logging_steps=10,                # Логирование каждые 10 шагов
            save_steps=500,                 # Сохранение модели каждые 500 шагов
            fp16=False,                       # Использовать fp16 для экономии памяти (если поддерживается GPU)
            gradient_accumulation_steps=2, # Увеличьте, если не хватает памяти
            optim="adafactor", # Оптимизатор для экономии памяти
            max_grad_norm=0.3, # Gradient clipping
            learning_rate=2e-5, # Learning Rate
            lr_scheduler_type="linear", #Добавил шедулер
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset, #Исправлено
            data_collator=lambda data: {"input_ids": torch.stack([f['input_ids'] for f in data]), #Переносим на GPU
                                       "attention_mask": torch.stack([f['attention_mask'] for f in data]), #Переносим на GPU
                                       "labels": torch.stack([f['labels'] for f in data])} #Переносим на GPU
        )

        # 5. Обучение модели
        trainer.train()

        # 6. Сохранение обученной модели
        model.save_pretrained("./trained_model")
        tokenizer.save_pretrained("./trained_model")

        await update.message.reply_text("Мяу! Переобучение завершено! Но это не значит, что я стала лучше! 😼")

    except Exception as e:
        logger.error(f"Ошибка переобучения: {str(e)}")
        await update.message.reply_text("Ой! Что-то пошло не так во время переобучения! 😿")

async def main():
    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("random", random_command))
    application.add_handler(CommandHandler("version", version_command))
    application.add_handler(CommandHandler("save", save_data))
    application.add_handler(CommandHandler("retrain", retrain_command))
    application.add_error_handler(error_handler) # Добавили обработчик ошибок
    
    await application.initialize()
    await application.start()
    await application.updater.start_polling(
        drop_pending_updates=True,
        timeout=30,
        poll_interval=2
    )
    
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        await application.updater.stop()
        await application.stop()
        await application.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Бот остановлен")
    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}"
