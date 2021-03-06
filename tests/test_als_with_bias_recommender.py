import sys
import os
sys.path.insert(0, os.path.join('.', 'answers'))
from answer import als_with_bias_recommender

def test_als_with_bias_recommender():
    a = als_with_bias_recommender(os.path.join('.', 'data', 'sample_movielens_ratings.txt'), 123)
    try:
        assert(abs(a-1.1210851900789454)<0.03)
    except:
        assert(abs(a-1.0409172341247102)<0.03)
