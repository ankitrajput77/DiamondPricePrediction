import pandas as pd
import os,sys
from src.pipelines import prediction_pipeline
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


print('Prediction for CSV file')
path = os.path.join('prediction_tries','test_data.csv')
df = pd.read_csv(path)
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

obj = prediction_pipeline.PredictPipeline()
y_pred = obj.predict(X)

y_pred = pd.DataFrame(y_pred)

path2 = os.path.join('prediction_tries','test_pred.csv')
y_pred.to_csv(path2, index=False, header=False)

print("R2 Score is :",r2_score(y_pred, y))

print('Prediction for Custom Data')


obj2 = prediction_pipeline.CustomData(0.5,62.1, 57.0, 5.05, 5.08, 3.14, 'Ideal', 'D', 'SI1')
df2 = obj2.get_data_as_dataframe()

obj_pred = prediction_pipeline.PredictPipeline()
y_pred = obj_pred.predict(df2)

print("Pred Value is :",y_pred)



