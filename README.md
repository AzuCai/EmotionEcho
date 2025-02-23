# EmotionEcho: Personalized Dialog Generator

## Project Overview
EmotionEcho is an innovative AI project that generates personalized, emotion-driven dialog responses based on user input. Built with a fine-tuned DistilGPT-2 model using LoRA adapters, it leverages the DailyDialog dataset to create natural, context-aware replies. Optimized for a 6GB GPU with 8-bit quantization, it includes a Gradio interface for easy interaction. Perfect for showcasing skills in NLP, model fine-tuning, and deployment during interviews.

## Features
- **Emotion-Driven Responses**: Generates replies reflecting user-selected emotions (happy, sad, angry, neutral).
- **Lightweight Model**: Uses DistilGPT-2 with LoRA for efficient fine-tuning on limited hardware (6GB VRAM).
- **Interactive UI**: Gradio interface for real-time testing and demonstration.
- **Optimized Performance**: Employs 8-bit quantization and mixed precision training to fit resource constraints.
- **Minimal Logging**: Training process streamlined with sparse logs and no progress bar clutter.

## LLM-Related Knowledge Points
- **Natural Language Processing (NLP)**: Processes and generates human-like dialog using the DailyDialog dataset.
- **Model Fine-Tuning**: Applies LoRA (Low-Rank Adaptation) to adapt a pre-trained DistilGPT-2 model efficiently.
- **Quantization**: Uses 8-bit quantization via `bitsandbytes` to reduce memory usage, enabling training on a 6GB GPU.
- **Parameter-Efficient Fine-Tuning (PEFT)**: Implements LoRA to fine-tune only a small subset of parameters, preserving model performance.
- **Prompt Engineering**: Structures inputs with `[Emotion:xxx]` tags to guide response generation.
- **Tokenization**: Utilizes Hugging Face’s tokenizer for text preprocessing and encoding.

## Environment Setup and Deployment

### Prerequisites
- **Hardware**: GPU with at least 6GB VRAM (e.g., NVIDIA GTX 1660).
- **Operating System**: Windows, Linux, or macOS (tested on Windows).
- **Python**: Version 3.9 or higher.

### Installation
1. **Clone the Repository**:
  ```bash
  git clone https://github.com/yourusername/EmotionEcho.git
  cd EmotionEcho
  ```
2. **Create Conda Environment (optional but recommended)**:
  ```bash
  conda create -n EmotionEcho python=3.10
  conda activate EmotionEcho
  ```
3. **Install Dependencies**:
  ```bash
  pip install transformers torch datasets gradio bitsandbytes accelerate peft
  ```
### Running the Project
1. **Train and Deploy**:

  Run the main script to fine-tune the model and launch the Gradio interface
  ```bash
  python main.py
  ```
  Training takes ~1-2 hours on a 6GB GPU (1 epoch, ~1900 steps). Logs appear every 500 steps.

2. **Interact**:
  After training, a browser window opens with the Gradio UI.
  
  Select an emotion (e.g., "happy") and enter a dialog history (e.g., "Hey, it’s a good day!") to generate a response.


### Troubleshooting
  - **Memory Issues**: Reduce per_device_train_batch_size to 2 in main.py if VRAM exceeds 6GB.

  - **Missing Modules**: Ensure all dependencies are installed; re-run the pip install command if errors occur.

  - **Custom Adjustments**: Edit logging_steps (e.g., to 100) in main.py for more frequent logs.

### Acknowledgments
  Built with Hugging Face Transformers and PEFT.
  Dataset: DailyDialog.
  Enjoy experimenting with EmotionEcho!


