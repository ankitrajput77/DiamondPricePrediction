from flask import Flask, render_template, request
from src.pipelines import prediction_pipeline
from src.logger import logging
from src.exception import CustomException


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict_dmd():
    try : 
        if request.method == 'GET':
            return render_template('home.html')
        
    except Exception as e :
        logging.info('Error Occured In Methods')

    else:
        try :
            data = prediction_pipeline.CustomData(
                carat=float(request.form.get('carat')),
                depth = float(request.form.get('depth')),
                table = float(request.form.get('table')),
                x = float(request.form.get('x')),
                y = float(request.form.get('y')),
                z = float(request.form.get('z')),
                cut = request.form.get('cut'),
                color= request.form.get('color'),
                clarity = request.form.get('clarity')
            )
            data_df = data.get_data_as_dataframe()
            predict_class_obj = prediction_pipeline.PredictPipeline()
            y_pred = predict_class_obj.predict(data_df)

            result=round(y_pred[0],2)

        except Exception as e :
            logging.info('Data Cannot be read from form')

        return render_template('home.html',result=result)
    

if __name__=="__main__":
    app.run(host='0.0.0.0')
