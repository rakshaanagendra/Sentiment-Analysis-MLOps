from src.train import train_and_evaluate

def test_training_runs():
    # Run with smaller params for speed
    train_and_evaluate(max_features=1000, ngram_range=(1, 1))
    # If no exception is raised → test passes
    assert True
