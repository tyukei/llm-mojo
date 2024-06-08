from python import Python
from time import time_function

fn chat_response() -> None:
    try:
        # モデルとトークナイザのロード
        var transformers = Python.import_module("transformers")
        var torch = Python.import_module("torch")
        var builtins = Python.import_module("builtins")
        var model_name = "gpt2"
        var tokenizer = transformers.GPT2Tokenizer.from_pretrained(model_name)
        var model = transformers.GPT2LMHeadModel.from_pretrained(model_name)

        # チャットボット用のプロンプト
        var user_input: String = builtins.input("You: ")
        var prompt: String = "The following is a conversation with an AI assistant. \n\nUser: " + user_input + "\nAI:"
        # トークン化
        var inputs = tokenizer(prompt, return_tensors="pt")
        var attention_mask = torch.ones_like(inputs.input_ids)

        # モデルを使用してテキストを生成
        var outputs = model.generate(inputs.input_ids, attention_mask=attention_mask, max_length=150, pad_token_id=tokenizer.eos_token_id)
        # 生成されたトークンをテキストにデコード
        var generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 最初の「AI:」の後のテキストを抽出
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