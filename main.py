import os
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import gradio as gr

# Disable symlinks warning for Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# --- Data Preprocessing Section ---
def preprocess_data():
    """
    Load and process the DailyDialog dataset into a format suitable for the model
    :return: Processed dataset split into train and test
    """
    # Load dataset from Hugging Face with trust_remote_code=True
    dataset = load_dataset("daily_dialog", split="train", trust_remote_code=True)

    processed_data = []
    for dialog in dataset["dialog"]:
        # dialog is a list of utterances, e.g., ["Hi", "Hello, how are you?"]
        for i in range(len(dialog) - 1):
            history = " | ".join(dialog[:i + 1])  # Dialog history separated by |
            response = dialog[i + 1]  # Target response
            # Simulate emotion tag (replace with real emotion labels if available)
            emotion = "happy" if "good" in history.lower() else "neutral"
            input_text = f"[Emotion:{emotion}] {history}"
            processed_data.append({"input": input_text, "output": response})

    # Convert to Hugging Face Dataset format and split into train/test
    dataset = Dataset.from_list(processed_data)
    return dataset.train_test_split(test_size=0.1)


# --- Model Loading and Fine-tuning Section ---
# Load tokenizer and model
model_name = "distilgpt2"  # Lightweight model suitable for 6GB VRAM
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token for distilgpt2
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,  # 8-bit quantization to reduce memory usage
    device_map="auto"  # Automatically map to GPU
)

# Add LoRA adapters for fine-tuning
lora_config = LoraConfig(
    r=16,  # Rank of the LoRA adapters
    lora_alpha=32,  # Scaling factor
    target_modules=["c_attn"],  # Target attention layers in distilgpt2
    lora_dropout=0.1,  # Dropout for regularization
    bias="none",  # Bias handling
    task_type="CAUSAL_LM"  # Task type for causal language modeling
)
model = get_peft_model(model, lora_config)  # Wrap model with LoRA


# Encoding function for dataset
def preprocess_function(examples):
    """
    Encode inputs and outputs into a format usable by the model
    :param examples: Data samples
    :return: Encoded data
    """
    inputs = tokenizer(examples["input"], truncation=True, max_length=128, padding="max_length")
    targets = tokenizer(examples["output"], truncation=True, max_length=128, padding="max_length")
    inputs["labels"] = targets["input_ids"]  # Set labels as target output
    return inputs


# Load and process dataset
dataset = preprocess_data()
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Define training arguments with reduced logging
training_args = TrainingArguments(
    output_dir="./results",  # Output directory
    num_train_epochs=1,  # Number of training epochs, set to 1 for quick testing
    per_device_train_batch_size=4,  # Training batch size
    per_device_eval_batch_size=4,  # Evaluation batch size
    warmup_steps=10,  # Warmup steps
    weight_decay=0.01,  # Weight decay
    logging_dir="./logs",  # Logging directory
    logging_steps=500,  # Log every 500 steps instead of every step
    save_steps=100,  # Model save frequency
    eval_strategy="steps",  # Evaluation strategy
    eval_steps=100,  # Evaluation frequency
    fp16=True,  # Mixed precision training to save memory
    disable_tqdm=True,  # Disable progress bar
)

# Define Trainer and start training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)
trainer.train()  # Start training


# --- Inference and Interface Section ---
def generate_response(emotion, history):
    """
    Generate a personalized response based on input
    :param emotion: Emotion tag
    :param history: Dialog history
    :return: Generated response
    """
    # Construct input prompt
    prompt = f"[Emotion:{emotion}] {history}"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,  # Limit generation length
        do_sample=True,  # Enable sampling for diversity
        temperature=0.7,  # Control randomness
        top_p=0.9  # Top-p sampling
    )

    # Decode and extract generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split(history)[-1].strip()


# Create Gradio interface
interface = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Dropdown(choices=["happy", "sad", "angry", "neutral"], label="Emotion"),  # Emotion selection
        gr.Textbox(lines=2, placeholder="Enter dialog history", label="Dialog History")  # Dialog history input
    ],
    outputs="text",  # Output as text
    title="EmotionEcho: Personalized Dialog Generator",  # Interface title
    description="Enter an emotion and dialog history to generate a personalized response!"  # Interface description
)

# Launch interface
interface.launch()