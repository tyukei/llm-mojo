import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import time

def chat_response():
    # Load the model and tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Prepare input text
    input_text = input("Enter the text to start with: ")
    prompt = "The following is a conversation with an AI assistant. \n\nUser: " + input_text + "\nAI:"
    start_time = time.time()

    # Tokenize the input text
    inputs = tokenizer(prompt, return_tensors="pt")

    # Set pad_token_id to eos_token_id to avoid unexpected behavior
    pad_token_id = tokenizer.eos_token_id

    # Generate text using the model
    outputs = model.generate(
        inputs.input_ids, 
        attention_mask=inputs.attention_mask,
        max_length=50,
        pad_token_id=pad_token_id
    )

    # Decode the generated tokens into text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text.split("AI:")[1].split("User:")[0].strip()
    print(response)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")

def main():
    chat_response()


if __name__ == "__main__":
    main()
