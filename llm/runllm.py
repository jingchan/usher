import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_name = "TheBloke/deepseek-llm-67b-chat-AWQ"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    offload_folder="./offload_folder",
).to(device)
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

messages = [
    {
        "role": "user",
        "content": "A person is stuck in a cave with a pail, 2 books (a physics textbook, and a Jane Eyre novel), a gun, and a 1x2 meter cloth, with no clothes or anything else. What is the plan to survive?",
    }
]
input_tensor = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(device)
outputs = model.generate(input_tensor.to(model.device), max_new_tokens=100)

result = tokenizer.decode(outputs[0][input_tensor.shape[1] :], skip_special_tokens=True)
print(result)
