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
