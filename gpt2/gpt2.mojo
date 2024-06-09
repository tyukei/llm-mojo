from python import Python


fn chat_response() -> None:
    try:
        # Load the model and tokenizer
        var transformers = Python.import_module("transformers")
        var torch = Python.import_module("torch")
        var builtins = Python.import_module("builtins")
        var model_name = "gpt2"
        var tokenizer = transformers.GPT2Tokenizer.from_pretrained(model_name)
        var model = transformers.GPT2LMHeadModel.from_pretrained(model_name)
        var time_function = Python.import_module("time")

        # Prepare input text
        var user_input: String = builtins.input("Enter the text to start with: ")
        var prompt: String = "The following is a conversation with an AI assistant. \n\nUser: " + user_input + "\nAI:"
        var start = time_function.time()

        # Tokenize the input text
        var inputs = tokenizer(prompt, return_tensors="pt")

        # Set pad_token_id to eos_token_id to avoid unexpected behavior
        var pad_token_id = tokenizer.eos_token_id

        # Generate text using the model
        var outputs = model.generate(
            inputs.input_ids, 
            attention_mask=inputs.attention_mask,
            max_length=50,
            pad_token_id=pad_token_id
        )

        # Decode the generated tokens into text
        var generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the response part of the generated text
        var response = generated_text.split("AI:")[1].split("User:")[0].strip()
        print("AI:", response)
        
        var end = time_function.time()
        var seconds = (end - start)
        print("Chatbot response took", seconds, "seconds to run")
    except e:
        print("An error occurred:", e)

fn main() -> None:
    chat_response()
