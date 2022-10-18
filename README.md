# SMS SPAM detection using NLP pipeline
Spam is unsolicited and unwanted messages sent electronically whose content may be malicious. The danger could be: exposure of pravicy, a fraud, or a virus,etc. Spam is a major problem for email users, and it is a growing problem for mobile phone users. The goal of this project is to build a model that can detect SMS spam. The dataset is from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection). The dataset contains 5572 SMS messages and is already labeled. The model is built using a NLP pipeline. 

## Dependencies
Requires [poetry](https://python-poetry.org/), whose installation instructions can be found [here](https://python-poetry.org/docs/#installing-with-the-official-installer). You also need to have right Python version installed through pyenv. Run `poetry install` to install all dependencies in the root directory of the project, where `poetry.lock` is located.

```
pyenv install 3.10.4
pyenv local 3.10.4
poetry env use 3.10.4
poetry install

```

## Usage
You can run the notebook in the root directory of the project, where `pyproject.toml` is located. 

```
poetry run jupyter notebook
```

## Analysis and Results
For the spam word cloud, we can see "free","cash",'prize","win" etc. those appealing words are used most commonly, meanwhile for the ham word cloud, those neutral words like "call","come","go","get" etc. are used mostly. To build a prediction classification model, the dataset is split into training and test sets with 80% and 20% of the data respectively. The model is built using an NLP pipeline. The pipeline includes: tokenization, stop word removal, and stemming. Experiments on different ways to do feature engineering, including TF-IDF, word2vec, and doc2vec. The model is trained in Random Forest Classifier. The model is evaluated using precision, recall and F1 score. The best model is Random Forest on TF-IDF, with 98% F1 score, 1.00% precision, 83% recall on the test dataset. This model is saved as a pickle file for future use.

![img](img/confusion_matrix.png)


## Project structure 
<pre>
.
├── data
│ └── spam.csv                  <- SMS spam dataset
├── scripts                     <- Experiment results folder
│   ├── util.py                 <- Functions to join,clean, tokenize text, word frequency
│   └── nlp_func.py             <- Functions to build NLP pipeline, train, test, and evaluate model
├── SMS_Spam_Classifier.ipynb   <- Notebook to load, explore, and preprocess data, create models
├── poetry.lock                 <- Poetry lock file for dependencies
├── pyproject.toml              <- Poetry project file for dependencies
├── .python-version          
├── .gitignore
└── README.md
 
</pre>

## Acknowledgements
Must give credit to [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) for the dataset and Udacity for the code review.

