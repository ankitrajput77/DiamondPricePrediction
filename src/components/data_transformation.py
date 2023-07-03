from sklearn.impute import SimpleImputer ## Handling Missing Values
from sklearn.preprocessing import StandardScaler # Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
from sklearn.pipeline import Pipeline # pipelines
from sklearn.compose import ColumnTransformer # merging
from src.exception import CustomException # exception handling
from src.logger import logging # logging function for loggers
from src.utils import save_object # util function for saving model pickle file 
from sklearn.preprocessing import FunctionTransformer # removing features
import pandas as pd
import numpy as np
import sys,os
from dataclasses import dataclass #dataclasses


## Data Transformation config
@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


# required function based on Feature engineering 
def drop_features(X):
    return X.drop(['x', 'y', 'z'], axis=1)


## Data Ingestionconfig class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()

    def preprocessor_object(self):
         
         try:
            logging.info('Data Transformation initiated')

            categorical_cols = ['cut', 'color','clarity']
            numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']
            
            # Define the custom ranking for each ordinal variable
            # As per domain expert
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            

            logging.info('Pipeline Initiated')

            
            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('remove_features', FunctionTransformer(drop_features, validate=False)),
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
                ]
            )
            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ('scaler',StandardScaler())
                ]

            )
            # merging pipelines
            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            
            logging.info('Pipeline Completed')
            return preprocessor

         except Exception as e:
            logging.info("Error in Data Trnasformation, Check data_transformation file")
            raise CustomException(e,sys)



    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head(3).to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head(3).to_string()}')

            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.preprocessor_object()
            drop_columns = ['price', 'id']

            ## features into independent and dependent features

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df = train_df['price']

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df['price']

            ## apply the transformation

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]


            # Util work to save model pickle
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info('Processsor pickle in created and saved')

            return(train_arr, test_arr)
        
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)


    
