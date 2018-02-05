import pandas as pd
from keras.models import load_model
from data_processing  import process_data, resample_dataframe, load_data_csv
import sys

MAX_NUM_WORDS = 50000
NUM_WORDS_SEQ = 100

#ARGS: train path modelload path modelsave path



def retrain_model(batch_X, batch_y, model_load_path, model_save_path):
    model = load_model(model_load_path)
    print(model.summary())
    print(model.train_on_batch(batch_X, batch_y))
    print(model.predict_classes(batch_X))
    model.save(model_save_path)


if __name__ == '__main__':
    cmd_arg = sys.argv
    assert len(cmd_arg) == 4
    train_path = cmd_arg[1]
    load_model_path = cmd_arg[3]

    save_model_path = cmd_arg[2]

    df = load_data_csv(train_path)

    X, y = process_data(df, MAX_NUM_WORDS, NUM_WORDS_SEQ, labels=True)
    retrain_model(X, y, load_model_path, save_model_path)


