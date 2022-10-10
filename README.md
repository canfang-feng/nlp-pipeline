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

## Run instructions:

Run the following commands in the project's **root directory** to process raw data, train and save model.

Step 1: run a pipeline to load profiles and activity data, clean data, create labels, extract features, and save the processed data to root directory for further ML modeling.
```
python process_data.py
```

Step 2: run ML pipeline to train classifier and save results to `experiments` folder (if not exist, it will be created). Results include model, feature importance, and model performance, prediction results on test set.
```
python train_model.py
```

Step 3: Try different models by changing the `models` dictionary in `main` function in `train_model.py` and run the script again. Compare the results in "experiments" folder and improve the model performance.

## Project structure 
<pre>
.
├── data
│ ├── activity                <- Folder of activity data
│ └── profiles                <- Folder of profiles data
├── experiments               <- Experiment results folder
│   ├── 20220911180846        <- Result of RandomForestClassifier
│   └── 20220911194611        <- Result of HistGradientBoostingClassifier
├── config.py                 <- Script to set up configurations and parameters
├── process_data.py           <- Script to process data
├── train_model.py            <- Script to train model
├── requirements.txt          <- The requirements file for reproducing the analysis environment
├── .gitignore
└── README.md
 
</pre>