# import Flask class from the flask module
from flask import Flask, request, render_template
import flask
import joblib
import numpy as np
import pickle
# Create Flask object to run
app = Flask(__name__)


@app.route('/')
def index():
    return flask.render_template("index.html")

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,4)
    loaded_model = joblib.load('./models/iris_model.pkl')
    result = int(loaded_model.predict(to_predict)[0])
    return result


@app.route('/predict', methods= ['POST'])
def predict():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        results = ValuePredictor(to_predict_list)

        return render_template('index.html', prediction_text='Predicted Iris Class {}'.format(results))
    else:
        return render_template('index.html')

if __name__ == "__main__":
    # Start Application
    app.debug=True
    app.run()
