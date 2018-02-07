import pandas as pd
from keras.models import load_model
# from data_processing  import process_data, load_data_csv

def retrain_model(batch_X, batch_y, model_path):
    model = load_model(model_path)
    print(model.summary())
    print(model.train_on_batch(batch_X, batch_y))
    print(model.predict_classes(batch_X))
    model.save(model_path)


if __name__ == '__main__':
    with open('../configurations.txt', 'r') as file:
        lines = file.readlines()[1:]
        user_samples_path = lines[11].split('=')[1].strip()

        save_model_path = lines[2].split('=')[1].strip()

        MAX_NUM_WORDS = int(lines[3].split('=')[1].strip())
        NUM_WORDS_SEQ = int(lines[4].split('=')[1].strip())


    df = load_data_csv(user_samples_path)

    X, y = process_data(df, MAX_NUM_WORDS, NUM_WORDS_SEQ, labels=True)
    print(X.shape)
    retrain_model(X, y, save_model_path)


