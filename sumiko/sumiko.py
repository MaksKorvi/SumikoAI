import os
import logging
import torch
import asyncio
import pandas as pd
from dotenv import load_dotenv
from typing import List
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType

# Загрузка переменных окружения
load_dotenv()
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("BOT_TOKEN")
MODEL_NAME = "sberbank-ai/rugpt3medium_based_on_gpt2"
MAX_RESPONSE_LENGTH = 200
TRAIN_STEPS = 5
BAD_WORDS = ["блять", "сука", "еблан", "хуй", "залупа", "уебище", "дура", "шлюха", "убежище", "ахуела", "мразь", "тварь", "мразота", "пидор", "ебанутая", "животное"]  # Замените на реальные стоп-слова

SYSTEM_PROMPT = """..."""  # Вставьте оригинальный системный промпт

# Инициализация модели и токенизатора
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Добавление специальных токенов
special_tokens = ['<SUMI>', '</SUMI>'] + ['♪', '(≧ω≦)', '(◕‿◕)', '(=^･ω･^=)', '♡']
tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
model.resize_token_embeddings(len(tokenizer))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def contains_profanity(text: str) -> bool:
    """Проверка на запрещенную лексику"""
    text_lower = text.lower()
    return any(bad_word in text_lower for bad_word in BAD_WORDS)

def filter_profanity(text: str) -> str:
    """Очистка текста от нецензурных слов"""
    words = text.split()
    cleaned_words = [word if not contains_profanity(word) else "*цензура*" for word in words]
    return " ".join(cleaned_words)

def format_prompt(user_input, history=None):
    """Форматирование промпта с историей"""
    system_part = f"<SUMI>{SYSTEM_PROMPT}</SUMI>\n"
    history_part = ""
    if history:
        for u, r in history[-3:]:
            history_part += f"👤: {u}\n🐱: {r}\n"
    return f"{system_part}{history_part}👤: {user_input}\n🐱:"

def generate_response(text, chat_history=None):
    """Генерация ответа с проверкой цензуры"""
    if contains_profanity(text):
        return "Мне разработчик запретил воспринимать такие слова! (◕︵◕)"
    
    prompt = format_prompt(text, chat_history)
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
    
    outputs = model.generate(
        inputs.input_ids,
        max_length=min(inputs.input_ids.shape[1] + MAX_RESPONSE_LENGTH, 1024),
        temperature=0.9,
        top_k=40,
        top_p=0.92,
        repetition_penalty=1.15,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1
    )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return postprocess_response(response)

def postprocess_response(text):
    """Постобработка ответа"""
    text = text.split("👤:")[0].strip()
    
    if contains_profanity(text):
        return "Ой, я не могу такое сказать! (=｀ω´=)"
    
    if not any(e in text for e in ['♪', '≧ω≦', '◕‿◕']):
        text += random.choice([' (≧ω≦)', ' (=^･ω･^=)', ' ♪'])
    
    return apply_cuteness(text[:MAX_RESPONSE_LENGTH])

def apply_cuteness(text):
    """Добавление милых опечаток"""
    replacements = {'спасибо': 'спс', 'пожалуйста': 'пжлста', 'привет': 'приветик'}
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик сообщений с цензурой"""
    user = update.effective_user
    message = update.message.text[:500]
    
    # Проверка на запрещенную лексику
    if contains_profanity(message):
        await update.message.reply_text("Мне разработчик запретил воспринимать такие слова! (◕︵◕)")
        return
    
    if not any(name in message.lower() for name in ["суми", "сумико"]):
        return
    
    user_data = context.user_data.setdefault('sumi', {
        'history': [],
        'train_buffer': [],
        'last_trained': None
    })
    
    response = generate_response(message, user_data['history'])
    user_data['history'].append((message, response))
    user_data['train_buffer'].append((message, response))
    
    user_data['history'] = user_data['history'][-10:]
    
    if len(user_data['train_buffer']) >= 5:
        await schedule_training(user_data)
    
    await update.message.reply_text(response)

async def schedule_training(user_data):
    """Планирование обучения с фильтрацией"""
    buffer = user_data['train_buffer'].copy()
    user_data['train_buffer'].clear()
    
    valid_data = [
        (q, a) for q, a in buffer 
        if len(a) > 15 
        and any(e in a for e in ['♪', '≧ω≦'])
        and not contains_profanity(q)
        and not contains_profanity(a)
    ]
    
    if valid_data:
        try:
            await asyncio.to_thread(fine_tune_model, valid_data)
            logger.info(f"Trained on {len(valid_data)} examples")
        except Exception as e:
            logger.error(f"Training error: {str(e)}")

def fine_tune_model(data):
    """Дообучение с LoRA"""
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    lora_model = get_peft_model(model, lora_config)
    
    dataset = Dataset.from_dict({
        "text": [format_prompt(q, []) + a for q, a in data]
    })
    dataset = dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=512), 
        batched=True
    )
    
    training_args = TrainingArguments(
        output_dir="./sumi-lora",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        learning_rate=1e-4,
        fp16=torch.cuda.is_available(),
        logging_steps=1
    )
    
    Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=dataset,
    ).train()

def main():
    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CommandHandler("reset", lambda u,c: c.user_data.clear()))
    application.run_polling()

if __name__ == "__main__":
    main()
