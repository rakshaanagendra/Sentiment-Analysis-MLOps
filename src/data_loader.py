import os   # Module to work with files and folders

def check_imdb_dataset():
    # Base path to your dataset
    base_path = "data/raw/aclImdb"

    # Define paths for positive and negative training reviews
    train_pos_path = os.path.join(base_path, "train/pos")
    train_neg_path = os.path.join(base_path, "train/neg")

    # List a few files inside each folder
    print("Some positive reviews:", os.listdir(train_pos_path)[:3])
    print("Some negative reviews:", os.listdir(train_neg_path)[:3])

    # Pick one positive review file and open it
    with open(os.path.join(train_pos_path, os.listdir(train_pos_path)[0]), 'r', encoding='utf-8') as f:
        text = f.read()

    # Print first 500 characters so we don’t flood the console
    print("\nSample review text (first 500 chars):\n", text[:500])

# This runs only when you execute the script directly
if __name__ == "__main__":
    check_imdb_dataset()
