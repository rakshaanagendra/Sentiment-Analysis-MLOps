import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download stopwords + tokenizer (run once)
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")

# Preprocessing function
def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()

    # 2. Remove HTML tags (e.g., <br />)
    text = re.sub(r"<.*?>", " ", text)

    # 3. Remove punctuation/numbers (keep only letters)
    text = re.sub(r"[^a-z\s]", "", text)

    # 4. Tokenize (split into words)
    words = word_tokenize(text)

    # 5. Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]

    return words

if __name__ == "__main__":
    sample = "I LOVED this movie! <br /> It was AMAZING, 10/10. Would watch again."
    print("Original:", sample)
    print("Processed:", preprocess_text(sample))
