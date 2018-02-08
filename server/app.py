from flask import Flask, request, render_template
import pandas as pd
from trainings.data_processing import process_data, load_data_csv
from trainings.lstm_train_new_sample import retrain_model
from numpy import argmax
import json

from keras.models import load_model


app = Flask(__name__)
user_data = ''
configs = {}
model = []
numb_samples = 0

def calc_sentiment(proba, w):
    p = proba[0].tolist()
    d = {0:-1, 1:1, 2:0}
    m = argmax(p)
    p.remove(p[m])
    if m == 2:
        return 0 + w * (p[1] - p[0])
    else:
        return d[m] - d[m] * (w * (p[0] + p[1]))




@app.before_first_request
def load_configs():
    global configs, model
    with open('../configurations.json') as json_data:
        configs = json.load(json_data)
        configs['resample'] = bool(configs['resample'])
    model = load_model(configs['model_path'])




@app.after_request
def retrain(response):
    global numb_samples
    if numb_samples >= configs['retrain_when']:
        df = load_data_csv(configs['user_samples_path'])

        X, y = process_data(df, configs['MAX_NUM_WORDS'], configs['NUM_WORDS_SEQ'], labels=True)
        retrain_model(X, y, configs['model_path'])
        numb_samples = 0
        with open(configs['user_samples_path'], 'w'):
            pass
    return response




@app.route('/', methods=['POST', 'GET'])
def predict():
    global user_data, model
    if request.method == 'POST':
        # try:
        if request.form['user-input-button'] == 'Submit text':
            user_data = request.form['user-input']
            print(user_data)

            df = pd.DataFrame([[None, ' ', user_data]], columns=['rating', 'title', 'review'])
            X, y = process_data(df, configs['MAX_NUM_WORDS'], configs['NUM_WORDS_SEQ'])
            p =  model.predict(X)
            print(p)
            sent = calc_sentiment(p, 0.1)
            return render_template('index.html', sentiment='Sentiment is ' + str(sent))
        else:
            labels = {'-1':0, '0':1, '1':2}

            user_label = request.form['user-input']
            try:
                label = labels[user_label]
            except KeyError:
                return render_template('index.html', note='Invalid label. Use -1 for neg, 0 for neutral, 1 for positive')
            with open(configs['user_samples_path'], 'a') as file:
                file.write(str(label) + ', ,' +  user_data + '\n')
            user_data = ''
            global numb_samples
            numb_samples += 1
            return render_template('index.html', note='Thanks for your cooperation')
    else:
        return render_template("index.html")



if __name__ == '__main__':
    app.run(debug=True)