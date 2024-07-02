import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Paths for models and tokenizers
original_model_path = "unsloth_llama3_8b"
finetuned_model_path = "output/llama3_finetune/checkpoint-60"
finetuned_tokenizer_path = "output/llama3_finetune/checkpoint-60"

# Function to load the original model and generate text
def generate_text_original(prompt):
    model, tokenizer, generator = None, None, None
    try:
        model = AutoModelForCausalLM.from_pretrained(original_model_path, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(original_model_path)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=200)
        result = generator(prompt, max_length=200, num_return_sequences=1)
        return result[0]['generated_text']
    finally:
        if model:
            del model
        if tokenizer:
            del tokenizer
        if generator:
            del generator
        torch.cuda.empty_cache()

# Function to load the fine-tuned model and generate text
def generate_text_finetuned(prompt):
    model, tokenizer, generator = None, None, None
    try:
        model = AutoModelForCausalLM.from_pretrained(finetuned_model_path, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(finetuned_tokenizer_path)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=200, do_sample=True, top_k=50)
        result = generator(prompt, max_length=200, num_return_sequences=1)
        return result[0]['generated_text']
    finally:
        if model:
            del model
        if tokenizer:
            del tokenizer
        if generator:
            del generator
        torch.cuda.empty_cache()

# Gradio interface
def gradio_interface(model_choice, prompt):
    if model_choice == "Original Model":
        return generate_text_original(prompt)
    elif model_choice == "Fine-Tuned Model":
        return generate_text_finetuned(prompt)

interface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Radio(["Original Model", "Fine-Tuned Model"], label="Model Choice"),
        gr.Textbox(lines=2, placeholder="Enter your prompt here...", label="Prompt")
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="OS-Verilog Copilot: FT-LLM with QLoRA and VeriGen Dataset using Unsloth",
    description="Select a model and enter a prompt to generate Verilog code."
)

if __name__ == "__main__":
    interface.launch()
