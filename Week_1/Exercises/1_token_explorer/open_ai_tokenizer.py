# Let's see real tokenization with tiktoken (OpenAI's tokenizer)
import tiktoken

encoder = tiktoken.get_encoding("cl100k_base")  # GPT-4's tokenizer

texts = [
    "Hello, how are you?",
    "Unbelievable",
    "I love eating biryani in Hyderabad",
    "मुझे बिरयानी बहुत पसंद है",  # Hindi
    "def calculate_gst(price, rate=0.18):",
]

for text in texts:
    tokens = encoder.encode(text)
    print(f"Text: {text}")
    print(f"Token IDs: {tokens}")
    print(f"Token count: {len(tokens)}")
    print(f"Tokens: {[encoder.decode([t]) for t in tokens]}")
    print("---")