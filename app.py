import pickle
import pandas as pd
import numpy as np
from flask import Flask,request
from flasgger import Swagger

app=Flask(__name__)
swagger = Swagger(app)
#When you pass __name__ as an argument to the Flask constructor, you are 
#telling Flask to use the current module as the starting point of your Flask application.

with open('note_classifier.pkl', 'rb') as file:
    classifier = pickle.load(file)
@app.route('/')
def Welcome():
    return "Welcome All"


@app.route('/predict',methods=["Get"])
def predict_note_authentication():
    """Lets Authenticate the Bank Note
    This is using docstrings for specifications.
    ---
    parameters:
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
        
    responses:
      200:
        description: Results
           
    """
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    predicted=classifier.predict([[variance,skewness,curtosis,entropy]])
    return "The Predicted Value is " + str(predicted[0])



@app.route('/predict_file',methods=["POST"])
def predict_note_file():
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    df_test=pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction=classifier.predict(df_test)
    return str(list(prediction))


if(__name__)=='__main__':
    app.run(host='0.0.0.0', port=8000)
