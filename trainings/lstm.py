import datetime
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras import losses
from data_processing  import process_data, resample_dataframe, load_data_csv
import json

with open('configurations.json') as json_data:
    configs = json.load(json_data)
    configs['resample'] =  bool(configs['resample'])



df = load_data_csv(configs['train_data_path'])

if configs['resample']:
    df = resample_dataframe(df)

print('Found {} training samples'.format(df.shape[0]))
print(df.head(5))

X_train, y_train = process_data(df, configs['MAX_NUM_WORDS'], configs['NUM_WORDS_SEQ'], labels=True)


print('Initializing model...')
model = Sequential()
model.add(Embedding(configs['MAX_NUM_WORDS'], configs['EMBEDDING_DIM'], input_length=configs['NUM_WORDS_SEQ']))
model.add(Dropout(0.2))
model.add(Conv1D(filters=configs['num_filters'], kernel_size=3, padding='same', activation='relu')) # filter_size - how many words to consider
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv1D(filters=configs['num_filters'], kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv1D(filters=configs['num_filters'], kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=25))

model.add(LSTM(configs['num_filters']))
# model.add(Flatten())
model.add(Dense(configs['num_filters'], activation='relu'))
model.add(Dense(3, activation='sigmoid'))
model.compile(loss=losses.sparse_categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
print(model.summary())



print('Training...')
model.fit(X_train, y_train, batch_size=configs['batch_size'], epochs=configs['epochs'])

df = load_data_csv(configs['test_data_path'])
if configs['resample']:

    df = resample_dataframe(df)


X_test, y_test = process_data(df, configs['MAX_NUM_WORDS'], configs['NUM_WORDS_SEQ'], labels=True)
acc = model.evaluate(X_test, y_test)

print('accuracy: {}'.format(acc))
with open('logs/log.txt', 'w') as file:
    file.write('{} accuracy: {}\n'.format(str(datetime.datetime.now()), acc))


model.save(configs['model_path'])