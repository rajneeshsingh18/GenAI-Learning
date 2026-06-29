import tiktoken

from dotenv import load_dotenv

load_dotenv()


# encoding_for_model()
# Auto-picks the right encoding. GPT-4o uses o200k_base — tiktoken handles this automatically.
def get_tokenizer(model = 'gpt-4o'):
    return tiktoken.encoding_for_model(model);


# encode()
# Converts text → list of integers. 'hello world' might become [15496, 995]. Each number = 1 token.
def count_tokens(text , model = 'gpt-4o'):
    encoder = get_tokenizer(model);
    token = encoder.encode(text)
    return len(token)
