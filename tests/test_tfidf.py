import sys
sys.path.append('..')
import unredactor 
def test_tfidf():
    
    
    text = ['I have a farm.', 'I love donuts', 'donuts are in the farm.']
    tfidf_list = unredactor.tfidf(text)
    print(tfidf_list) 

    assert type(tfidf_list) == dict
