# Tokenizer

This repository implements a tokenizer similar to (but a lot simpler than) 
the GPT tokenizer and contains the text used to train the tokenizer and the
merges that resulted from this process. Current implementation does not deal
with special tokens.


## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repository.git
   
2. Install dependencies:
    ```bash
    pip install -r requirements.txt

3. Use the tokenizer class:
    ```python
    from tokenizer.tokenizer.encoding_decoding import Tokenizer
    
    text = 'Trapped inside this octavarium'
    tokenizer = Tokenizer()
    encoded_text = tokenizer.encode(text)
    decoded_text = tokenizer.decode(encoded_text)