import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from recommender import MovieRecommender

def test_recommendations_and_search():
    movies = pd.DataFrame({'movieId':[1,2,3,4], 'title':['Alpha (2000)','Beta (2001)','Gamma (2002)','Delta (2003)'], 'genres':['Comedy|Drama','Comedy','Action','Drama']})
    ratings = pd.DataFrame({'userId':[1,1,2,2,3,3], 'movieId':[1,2,2,3,1,4], 'rating':[5.,4.,4.,5.,4.,3.]})
    tags = pd.DataFrame({'userId':[1], 'movieId':[1], 'tag':['funny']})
    model = MovieRecommender(movies, ratings, tags)
    recs = model.recommend(user_id=1, liked_movie_ids=[1], limit=2)
    assert len(recs) > 0
    assert 'score' in recs.columns
    assert model.search('Alpha').iloc[0]['title'] == 'Alpha (2000)'
