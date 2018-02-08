import pandas as pd
from keras.models import load_model
# from data_processing  import process_data, load_data_csv

def retrain_model(batch_X, batch_y, model_path):
    model = load_model(model_path)
    print(model.summary())
    print(model.train_on_batch(batch_X, batch_y))
    print(model.predict_classes(batch_X))
    model.save(model_path)




