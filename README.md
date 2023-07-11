# The Unredactor
# Author Nasri Binsaleh

Original repo name: cs5293sp22-project3

#### This is the unredactor where the program can predict the what would be the most likely names that were in the texts before they were redacted. 

#### Heads up! The program can take some time to run, please don't give up on waiting! 

# How to install, directions on how to use the code, and some example of how to run.

## First, installation, simply clone the github ripository to your machine.
This github repository can be cloned using the following command:-  
    ```git clone <git repository link>``` the link in this case is https://github.com/bins0000/cs5293sp22-project3
    
## Activating pipenv in the project directory
inside the project directory `cs5293sp22-project3` run the following command to create a pip environment
```
pipenv install .
```
while creating a pip environment, the dependencies should autumatically be installed. 
## Prerequisites
      [[source]]
      url = "https://pypi.org/simple"
      verify_ssl = true
      name = "pypi"

      [packages]
      nltk = "*"
      numpy = "*"
      sklearn = "*"
      pandas = "*"
      pytest = "*"

      [dev-packages]

      [requires]
      python_version = "3.10"
If the packages above were not installed when the environment was created, you may manualy install each package in the pre-requisites above using `pipenv install <package>`

## Directories
    cs5293sp22-project2
      .
      ├── COLLABORATORS
      ├── LICENSE
      ├── Pipfile
      ├── Pipfile.lock
      ├── README.md
      ├── setup.cfg
      ├── setup.py
      ├── tests
      │   ├── test_get_features.py
      │   └── test_tfidf.py
      ├── unredactor.py
      └── unredactor.tsv
The program to run is ```unredactor.py``` in the main directory cs5293sp22-project3

## How to run
You can call `unredactor.py` in the commandline to run the program by using the following command as an example:- 

    pipenv run python unredactor.py  

By putting the class-collaborated dataset (unredactor.tsv) in the root directory cs5293sp22-project3, the program (unredactor.py) will automatically read the dataset and categorize them into training, testing, and validation set. Therefore, you must put unredactor.tsv in the root directory. 

# Functions in this program

## The Global Main
Here, the global variables and objects are created to be used in the main and other functions. The training, testing, and validation data are read from unredactor.tsv into the lists accordingly. For example, the traning dataset will be read into collab_train_x list, and the traning labels will be read into collab_train_y. Similarly with the test and validation dataset. A little bit of data cleaning was done as well because some of the students put sentences with more than one redacted items. 
The tfidf dictionary is also created here by calling tfidf function described below. Then the main function is called.

## tfidf(dataset)
this function takes in a list of data and compute the tfidf value for each of the words in the sentences. The dataset that is passed into the function is vectorized using tfidfvectorizer, and the tfidf score of each words were stored to be used as one of the features later on in get_features function. 

## get_features(text)
This is rather important in determining the performance of our predictive model. In this get_features function, a string of text is passed into the function and the features of that text is extracted. when passing in the train data with names being redacted, the function recognizes the length of the name by counting the number of block unicode that appears in the text. The length of the redacted name is one of the features. 
Then, around the name, are the words before and after the name which are also analyzed using their tfidf values. The tfidf values of the words before and after the name are stored as another two features for this text. Next, the spaces within the redacted names are also counted and stored as another feature. 
Another feature is the length of the text which is counted by the number of character in the given text. Lastly, the error value is also beign used as a feature here as many of the data in this set failed to be recognized as a redacted sentences. 

## Main Function
The main function drives the program. In the main function, a random forest model is being developed. First, the the training data are iterated through and their features are extracted using the get_features function. Then a random forest classifier is modeled by vectorizing the features extracted earlier with DictVectorizer. The training labels are also converted into a numpy array to match the features. Then the features and the labels are fit into the random forest model. 
Once the model is trained with the training dataset, then the test in testing set is done. model.predict is used to predict the names that are redacted. Then with the results obtained, they can be compared with the test labels for the accuracy, precision, recall, and f1-score. 
Using metrics.classification_report from sci-kit learn, the accuracy, precision, recall, and f1-score can be computed. The sample output can be seen below. 
                              precision    recall  f1-score   support
                  A.R. Rahman       0.00      0.00      0.00         0
                         Adam       0.00      0.00      0.00         3
                   Adam Beach       0.00      0.00      0.00         1
                 Adam Sandler       0.00      0.00      0.00         2
                Adrian Pasdar       0.00      0.00      0.00         0
                       spacek       0.00      0.00      0.00         0

                     accuracy                           0.01       563
                    macro avg       0.00      0.00      0.00       563
                 weighted avg       0.01      0.01      0.01       563

# Assumptions & Bugs
## Assumptions
#### Random Forest is the one
- Other models were also tried, but Random Forest are the one with the highest accuracy score so far. So, I assumed that Random Forest might be the best model to use in this case. In reality, there might be a better model to be used as a classifier. 
#### The amount of trainging dataset
- Since it is very time and resource consuming to create a model with a hugh anmount of data, the model in this program then uses the minimal amount of data. 
#### Assuming that the ratio of train/test/validation set is reasonalble
Since the ratio of the dataset provided is 50:30:10 for train:validation:test. We would assume that this is a good amount of training data. 

## Bugs
#### VM intance size
- the size of VM is not large enough, so the dataset is forced to be minimized. Only less than a thousand data is used for training sue to the kernel being killed if exceeds the limit. 
#### Accuracy
- With such low amount of training data, the results are definitely not going to be good. On top of that, low amount of data is not enough to predict a hugh amount of names. 


# Test
``` 
tests
   ├── test_get_features.py
   └── test_tfidf.py
```
As can be seen from the trees above, a couple of tests were done to check if a particular component is working.  
  
### test_get_features.py
This test was done to check if the get_features function is working correctly. The test was done to check if the function is returing the desired element including whether the features are extracted correctly and reasonably and if the return object is in the correct format. 

### test_tfidf.py
And this test is a test for our tfidf dictionary. It tests if the tfidf values are computed and stored correctly in a dictionary format. 

# Outputs
The output of this program shows you the most likely name that might have been redacted and its corresponding redacted text.
```
     Predicted Names                                      Redacted Text
0      Dennis Hopper  The mentor this time is played perfectly by ██...
1           Ms Mehta  While his scenes with the local love interest ...
2      Elenor Parker  Well, I have to disagree with ██████████████ o...
3       Anthony Wong  Director ████████████ offers endless laughters...
4     Kim Longinotto  "It will be the death of ███████████████"" Fas...
..               ...                                                ...
558    Dennis Hopper  I always thought of her as more matronly, but ...
559      Suzy Parker  The only thing worthwhile is the top-notch ███...
560  Jayne Mansfield  Supporting cast includes ██████████████, Mary ...
561          Lakshya  This fanciful horror flick has ███████ Price p...
562        Charlotte  Supporting cast includes Patrick O'Neal, Mary ...
```


## Related Links
- https://oudatalab.com/cs5293sp22/projects/project3
- https://scikit-learn.org/stable/supervised_learning.html#supervised-learning
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- https://www.nltk.org/
