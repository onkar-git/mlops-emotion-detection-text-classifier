from flask import Flask, render_template,request
from preprocessing_utility import normalize_text
import dagshub
import mlflow
import pickle

app =Flask(__name__)

#load model from mlflow model registry
mlflow.set_tracking_uri('https://dagshub.com/onkar-git/mlops-emotion-detection-text-classifier.mlflow')
dagshub.init(repo_owner='onkar-git', repo_name='mlops-emotion-detection-text-classifier', mlflow=True)

model_name = "my_model"
model_version = 5

model_uri = f'models:/{model_name}/{model_version}'

model = mlflow.pyfunc.load_model(model_uri)

vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))



@app.route('/')
def home():
    return render_template('index.html',result=None)

@app.route('/predict',methods=['POST'])
def predict():
    text=request.form['text']

    #load model from model regstry


    # clean 
    text = normalize_text(text)

    features = vectorizer.transform([text])
     
    #BOW

    result = model.predict(features)

    return render_template('index.html',result=str(result[0]))
    


    #prediction




    #show  predictio

app.run(debug=True,port=8000)