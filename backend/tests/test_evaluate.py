import pytest
from backend.scripts.evaluate import recall_at_k, mean_reciprocal_rank

def test_recall_at_k():
    ranks = [1, 2, float('inf'), 5, 10, float('inf')]
    
    assert recall_at_k([], 5) == 0.0
    
    # K=1: only rank=1 counts
    assert recall_at_k(ranks, 1) == 1/6
    
    # K=5: ranks 1, 2, 5 count
    assert recall_at_k(ranks, 5) == 3/6
    
    # K=10: ranks 1, 2, 5, 10 count
    assert recall_at_k(ranks, 10) == 4/6

def test_mean_reciprocal_rank():
    ranks = [1, 2, float('inf'), 4]
    
    assert mean_reciprocal_rank([]) == 0.0
    
    # MRR = (1/1 + 1/2 + 0 + 1/4) / 4 = (1 + 0.5 + 0.25) / 4 = 1.75 / 4 = 0.4375
    assert mean_reciprocal_rank(ranks) == 0.4375
import os
import json
from unittest.mock import patch
from backend.core.config import settings

@pytest.mark.slow
def test_evaluate_smoke_test(tmp_path):
    if not os.path.exists(settings.IMAGES_INDEX_FILE) or not os.path.exists(settings.CAPTIONS_INDEX_FILE):
        pytest.skip("Real index files not found, skipping smoke test.")
    
    test_args = [
        "evaluate.py",
        "--sample-size", "5",
        "--num-seeds", "1",
        "--gallery-size", "50",
        "--output-dir", str(tmp_path)
    ]
    with patch("sys.argv", test_args):
        import backend.scripts.evaluate as evaluate
        evaluate.main()
        
    results_file = tmp_path / "results_50.json"
    assert results_file.exists(), "Results file was not created in output-dir"
    
    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert "meta" in data
    meta = data["meta"]
    
    expected_keys = ["date", "model_id", "versions", "seeds", "sample_size", "gallery_size"]
    for key in expected_keys:
        assert key in meta, f"Missing key {key} in meta"
        
    assert meta["gallery_size"] == 50
