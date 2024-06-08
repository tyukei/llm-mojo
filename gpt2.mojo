from python import Python

fn main() raises:
    # モデルとトークナイザのロード
    var transformers = Python.import_module("transformers")
    var torch = Python.import_module("torch")
    var builtins = Python.import_module("builtins")

    var model_name = "gpt2"
    var tokenizer = transformers.GPT2Tokenizer.from_pretrained(model_name)
    var model = transformers.GPT2LMHeadModel.from_pretrained(model_name)

    # チャットボット用のプロンプト
    print("You: ")
    var user_input: String = builtins.input()

    var prompt: String = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nUser: " + user_input + "\nAI:"

    # トークン化
    var inputs = tokenizer(prompt, return_tensors="pt")
    var attention_mask = torch.ones_like(inputs.input_ids)

    # モデルを使用してテキストを生成
    var outputs = model.generate(inputs.input_ids, attention_mask=attention_mask, max_length=150, pad_token_id=tokenizer.eos_token_id)

    # 生成されたトークンをテキストにデコード
    var generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 最初のプロンプトを除去して応答のみを表示
    var response = generated_text.split("AI:")[1].strip()

    print("AI:", response)

