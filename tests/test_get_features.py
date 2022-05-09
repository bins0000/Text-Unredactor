import sys
sys.path.append('..')
import unredactor
import re
def test_get_features():

    text_list = ['I have a farm.', 'I love donuts', 'donuts are in the farm.']
    tfidf_dict = unredactor.tfidf(text_list)

    text = "Today, █████ went to school in the morning."
    block = '\u2588'
    features_list = []
    l = len(text)
    left_of_name = r'(\w*|\W)\s*'+ block + r'+\s*' + block + r'+\s*' + block + r'+\s*' + block + r'+\s*'+ block + r'+'
    name = block + r'+\s*' + block + r'+\s*' + block + r'+\s*' + block + r'+\s*' + block + r'+'
    right_of_name = block + r'+\s*' + block + r'+\s*' + block + r'+\s*' + block + r'+\s*' + block +r'+\s*(\W{0,1}\w*)'
    name_redacted = re.findall(name, text)
    left = re.findall(left_of_name, text)
    right = re.findall(right_of_name, text)
    if len(name_redacted) == 0:

        '''
        index = collab_train_x.index(text)
        collab_train_x.pop(index)
        collab_train_y.pop(index)
        '''
        feature_dict = {}
        feature_dict['error'] = 1
        feature_dict['name_length'] = 0
        feature_dict['spaces'] = 0
        feature_dict['left_name'] = 0
        feature_dict['right_name'] = 0
        feature_dict['len_chars'] = 1
        features_list.append(feature_dict)

    else:
        for i in range(len(name_redacted)):
            feature_dict = {}

            if left[i] in tfidf_dict.keys():
                tfidf_left = tfidf_dict[left[i]]
            else:
                tfidf_left = 0

            if right[i] in tfidf_dict.keys():
                tfidf_right = tfidf_dict[right[i]]
            else:
                tfidf_right = 0

            feature_dict['name_length'] = len(name_redacted[i])
            feature_dict['spaces'] = name_redacted[i].count(' ')
            feature_dict['left_name'] = tfidf_left
            feature_dict['right_name'] = tfidf_right
            feature_dict['len_chars'] = 1
            feature_dict['error'] = 0
            features_list.append(feature_dict)

    assert type(features_list) == list
    assert feature_dict['name_length'] == 5
