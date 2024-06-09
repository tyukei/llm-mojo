from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "EleutherAI/gpt-j-6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

input_text = "The following is a conversation with an AI assistant.\n\nUser: Who was born in the city of Kolkata?\nAI:"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate text
outputs = model.generate(inputs.input_ids, max_length=50, pad_token_id=tokenizer.eos_token_id)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)
