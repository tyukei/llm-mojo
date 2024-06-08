from python import Python
from time import time_function

fn chat_response() -> None:
    try:
        # モデルとトークナイザのロード
        var transformers = Python.import_module("transformers")
        var torch = Python.import_module("torch")
        var builtins = Python.import_module("builtins")
        var model_name = "t5-small"
        var tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
        var model = transformers.T5ForConditionalGeneration.from_pretrained(model_name)

        # チャットボット用のプロンプト
        var user_input: String = builtins.input("You: ")
        var prompt: String = "The following is a conversation with an AI assistant. \n\nUser: " + user_input + "\nAI:"

        
        # トークン化
        var inputs = tokenizer(prompt, return_tensors="pt")

        # モデルを使用してテキストを生成
        var outputs = model.generate(inputs.input_ids, max_length=150)
        # 生成されたトークンをテキストにデコード
        var generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        var response = generated_text.split("AI:")[1].split("User:")[0].strip()
        print("AI:", response)
    except e:
        print("An error occurred:", e)

fn main() -> None:
    try:
        var time_function = Python.import_module("time")
        var start = time_function.time()
        chat_response()
        var end = time_function.time()
        var seconds = (end - start)
        print("Chatbot response took", seconds, "seconds to run")
    except e:
        print("An error occurred:", e)

