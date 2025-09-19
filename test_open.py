file_path = "data/raw/acImdb/IMDB_Dataset.csv"

with open(file_path, "r", encoding="utf-8") as f:
    print("First 100 chars:", f.read(100))
