from flask import Flask, render_template, request
from src.pipelines import prediction_pipeline


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict_dmd():
    if request.method == 'GET':
        return render_template('home.html')
    else:
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

        return render_template('home.html',result=result)
    

if __name__=="__main__":
    print('http://127.0.0.1:5000/')
    app.run(host='0.0.0.0')