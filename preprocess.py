from nltk.tokenize import RegexpTokenizer

def clean_text(text, tokenizer=RegexpTokenizer(r"[a-z]+")):
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    cleaned_text = " ".join(tokens)
    return cleaned_text