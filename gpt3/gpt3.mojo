from python import Python

fn main() raises:
    var openai = Python.import_module("openai")
    var os = Python.import_module("os")
    var dot_env = Python.import_module("dotenv")
    dot_env.load_dotenv()
    var api_key =  os.getenv('OPENAI_KEY')
    var client = openai.OpenAI(api_key=api_key)
    var dictionary = Python.dict()
    dictionary["role"] = "user"
    var builtins = Python.import_module("builtins")
    dictionary["content"] = builtins.input("Enter your message: ")
    
    # Explicitly create a list with proper conversion
    var messages = Python.list()
    messages.append(dictionary)
    
    var completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    print(completion.choices[0].message.content)


