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


MODEL_COSTS = {
    'gpt-4o':      {'input': 5.00,  'output': 15.00},
    'gpt-4o-mini': {'input': 0.15,  'output': 0.60}
}

# Context window
# Max tokens a model can 'see' at once. GPT-4o = 128K tokens. Exceed it → model can't process your text.
CONTEXT_WINDOWS = {'gpt-4o': 128000, 'gpt-4o-mini': 128000}



def calculate_cost(token_count, model='gpt-4o', direction='input'):
    price_per_million = MODEL_COSTS[model][direction]
    return (token_count / 1_000_000) * price_per_million


def get_context_info(token_count, model='gpt-4o'):
    max_t = CONTEXT_WINDOWS[model]
    pct = (token_count / max_t) * 100
    return {'max_tokens': max_t, 'percentage_used': round(pct, 2),
            'fits': token_count <= max_t, 'warning': pct > 80}


def main():
    print('\n' + '='*50)
    print('TOKEN EXPLORER')
    print('='*50)
    print('Enter text, then press Ctrl+D (or Ctrl+Z on Windows)\n')
    
    lines = []
    try:
        while True:
            lines.append(input())
    except EOFError:
        pass  # Ctrl+D ends input
    
    text = '\n'.join(lines)
    
    for model in ['gpt-4o', 'gpt-4o-mini']:
        count = count_tokens(text, model)
        cost  = calculate_cost(count, model, 'input')
        info  = get_context_info(count, model)
        print(f'  {model}: {count:,} tokens | ${cost:.6f} | {info["percentage_used"]}% of context')

if __name__ == '__main__':
    main()