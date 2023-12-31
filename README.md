# Machine Learning Project

Diamond price prediction is the process of estimating the value or cost of a diamond based on various factors and characteristics. It involves using data analysis, statistical modeling, and machine learning techniques to predict the market price or worth of a diamond.

Diamonds are precious gemstones that are evaluated based on their unique features, known as the "Four Cs": carat, cut, color, and clarity. These factors, along with additional aspects such as depth and table, play a crucial role in determining a diamond's value.

# Requirements
- [sklearn](https://scikit-learn.org/stable/)
- [pandas](https://www.w3schools.com/python/pandas/default.asp)
- [flask](https://flask.palletsprojects.com/en/2.3.x/)
- [seaborn](https://seaborn.pydata.org/)
- [python](https://www.python.org/)

# Installation and Usage

To install webApp, follow these steps:

Environment Setup
```
conda create -p venv python==3.8
```
```
conda activate venv/
```

1. Clone the repository:
```
git clone https://github.com/ankitrajput77/DiamondPricePrediction.git
```

2. Navigate to the project directory:
```
cd DiamondPricePrediction
```
3. Install dependencies:
```
pip install -r requirements.txt
```
4. Set up :
```
python application.py
```
5. Start the server:
```
http://127.0.0.1:5000/
```

# Notebook
Used dataset is present on kaggle 
- Use this notebook to access : [Kaggle Notebook](https://www.kaggle.com/code/ankitrajput77/eda-modeltrain-pipeline)
- LinkedIn Post of Notebook (you can download notebook pdf) : [LinkedIn Post](https://www.linkedin.com/feed/update/urn:li:activity:7081327130298527744?utm_source=share&utm_medium=member_desktop)
# In Details
```
├──  artifacts                    - here's the model pickle and dataset files.
│    └── preprocessor.pkl  
│    └── model.pkl
│    └── raw.csv
│    └── train.csv
│    └── test.csv
│
│
├──  Logs  
│    └── time_format.log          - here's the specific run log files.
│ 
│
├──  notebooks  
│    └── EDA.ipynb                        - here's the EDA notebook.
│    └── Model-training.ipynb             - here's the model training notebook.
│    └── data 		                        - here's the folder for dataset.
│          └── gemstone.csv               - here's the dataset present about diamond details.
│
│
├──  prediction_tries
│   ├── prediction.py                     - code to predict the price for test_data.
│   └── test_data.csv                     - here's the test_data.
│   └── test_pred.csv                     - here's the predicted values for test_data(it will generate after running prediction.py).
│
│
│
├── src                                   - The "src" folder, short for "source".
│   └── exception.py                      - Exception handling.
│   └── logger.py                         - log file handling.
│   └── utils.py                          - util functions.
│   └── components
│          └── data_ingestion.py          - code for data ingestion.
│          └── data_transformation.py     - code for data transformation.
│          └── model_trainer.py           - code for model training.
│   └── pipelines
│          └── prediction_pipeline.py     - code for model prediction 
│          └── training_pipeline.py       - code for training of model 
│
│
│
├── static                                - this folder contains frontend css files.
│   └── css
│        └── styles.css 
│   └── images
│        └── github_logo
│        └── kaggle_logo
│
│
├── templates                             - this folder contains html files.
│   └── home.html
│   
│ 
│ 
├── application.py                        - Code for webapp running.
│					
└──setup.py                               - project's metadata and configuration details
```
# InAction
<img width="855" alt="image" src="https://github.com/ankitrajput77/DiamondPricePrediction/assets/113281225/790895da-0916-4361-9fc1-60de5308c03b">


# Contributing
Any kind of enhancement or contribution is welcomed.

## Contact
If you have any questions, feedback, or suggestions, feel free to reach out to us at [rajputankit72106@gmail.com](mailto:rajputankit72106@gmail.com).
