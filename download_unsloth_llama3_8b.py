from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "unsloth/llama-3-8b-bnb-4bit"

# Download and save the model
model = AutoModelForCausalLM.from_pretrained(model_id)
model.save_pretrained("./unsloth_llama3_8b")

# Download and save the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained("./unsloth_llama3_8b")

print("Model and tokenizer downloaded successfully.")
