from src.build_dataset import load_imdb_dataset

def test_load_dataset():
    df = load_imdb_dataset()
    # Check it's not empty
    assert not df.empty
    # Check it has expected columns
    assert "text" in df.columns
    assert "label" in df.columns
