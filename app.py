from wsgiref import simple_server

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)
from logger.LoggerClass import LoggerFileClass

logger = LoggerFileClass("ui_page")
model = joblib.load('models/XGBoost_Regressor_model.pkl')  # loading the saved XGBoost_regressor model


@app.route("/")
def home():
    logger.add_info_log("class app, load home page")
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    For rendering results on HTML GUI

    """
    if request.method == "POST":
        # ['age', 'cement', 'water', 'fly_ash', 'superplasticizer', 'blast_furnace_slag']
        f_list = [request.form.get('age'), request.form.get('cement'), request.form.get('water'),
                  request.form.get('fa'),
                  request.form.get('sp'), request.form.get('bfs')]  # list of inputs

        # logging operation
        logger.add_info_log(f"class app, predict function : Age (in days): {f_list[0]}, Cement (in kg): {f_list[1]},"
                            f"Water (in kg): {f_list[2]}, Fly ash (in kg): {f_list[3]},"
                            f"Superplasticizer (in kg): {f_list[4]}, Blast furnace slag (in kg): {f_list[5]}")

        final_features = np.array(f_list).reshape(-1, 6)
        df = pd.DataFrame(final_features)

        prediction = model.predict(df)
        result = "%.2f" % round(prediction[0], 2)

        # logging operation
        #         logging.info(f"The Predicted Concrete Compressive strength is {result} MPa")

        #         logging.info("Prediction getting posted to the web page.")
        prediction = f"The Concrete compressive strength is {result} MPa"



        return render_template('index.html',
                  prediction_text=f"The Concrete compressive strength is {result} MPa")
import os
port = int(os.getenv("PORT",5001))
if __name__ == "__main__":
    #app.run(debug=True)
    host = '0.0.0.0'
    httpd = simple_server.make_server(host=host,port=port,app=app)
    httpd.serve_forever()

