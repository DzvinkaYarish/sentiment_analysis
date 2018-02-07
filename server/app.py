from flask import Flask, request, render_template
import pandas as pd
from trainings.data_processing import process_data, load_data_csv
from trainings.lstm_train_new_sample import retrain_model

from keras.models import load_model


app = Flask(__name__)
user_data = ''
model_path = ''
user_samples_path = ''
retrain_when = ''
numb_samples = 0
MAX_NUM_WORDS = 0
NUM_WORDS_SEQ = 0


# def calc_sentiment(proba):




@app.before_first_request
def load_configs():
    with open('../configurations.txt', 'r') as file:
        lines = file.readlines()[1:]
        global  MAX_NUM_WORDS, NUM_WORDS_SEQ, model_path, user_samples_path, retrain_when
        MAX_NUM_WORDS = int(lines[3].split('=')[1].strip())
        NUM_WORDS_SEQ = int(lines[4].split('=')[1].strip())
        model_path = lines[2].split('=')[1].strip()
        user_samples_path = lines[11].split('=')[1].strip()
        retrain_when = int(lines[12].split('=')[1].strip())


@app.after_request
def retrain(response):
    global numb_samples
    if numb_samples >= retrain_when:
        df = load_data_csv(user_samples_path)

        X, y = process_data(df, MAX_NUM_WORDS, NUM_WORDS_SEQ, labels=True)
        retrain_model(X, y, model_path)
        numb_samples = 0
        with open(user_samples_path, "w"):
            pass
    return response




@app.route('/', methods=['POST', 'GET'])
def predict():
    global user_data
    if request.method == 'POST':
        # try:
        if request.form['user-input-button'] == 'Submit text':
            user_data = request.form['user-input']
            print(user_data)

            df = pd.DataFrame([[None, ' ', user_data]], columns=['rating', 'title', 'review'])
            X, y = process_data(df, MAX_NUM_WORDS, NUM_WORDS_SEQ)
            model = load_model(model_path)
            prediction = model.predict_classes(X)
            p =  model.predict(X)
            print(p)
            return render_template('index.html', sentiment='Sentiment is ' + str(prediction[0]))
        else:
            labels = {'-1':0, '0':1, '1':2}
            global numb_samples
            numb_samples += 1
            user_label = request.form['user-input']
            try:
                label = labels[user_label]
            except KeyError:
                return render_template('index.html', note='Invalid label. Use -1 for neg, 0 for neutral, 1 for positive')
            with open(user_samples_path, 'a') as file:
                file.write(str(label) + ', ,' +  user_data + '\n')
            user_data = ''
            return render_template('index.html', note='Thanks for your cooperation')
    else:
        return render_template("index.html")






if __name__ == '__main__':
    app.run(debug=True)