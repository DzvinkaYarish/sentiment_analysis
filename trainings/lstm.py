import datetime
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras import losses
from data_processing  import process_data, resample_dataframe, load_data_csv


with open('configurations.txt', 'r') as file:
    lines = file.readlines()[1:]
    train_path = lines[0].split('=')[1].strip()
    test_path = lines[1].split('=')[1].strip()
    save_model_path = lines[2].split('=')[1].strip()

    MAX_NUM_WORDS = int(lines[3].split('=')[1].strip())
    NUM_WORDS_SEQ = int(lines[4].split('=')[1].strip())
    EMBEDDING_DIM = int(lines[5].split('=')[1].strip())

    num_filters = int(lines[6].split('=')[1].strip())
    kernel_size = int(lines[7].split('=')[1].strip())
    epochs = int(lines[8].split('=')[1].strip())
    batch_size = int(lines[9].split('=')[1].strip())
    resample = bool(lines[10].split('=')[1].strip())

df = load_data_csv(train_path)

if resample:
    df = resample_dataframe(df)

print('Found {} training samples'.format(df.shape[0]))
print(df.head(5))

X_train, y_train = process_data(df, MAX_NUM_WORDS, NUM_WORDS_SEQ, labels=True)


print('Initializing model...')
model = Sequential()
model.add(Embedding(MAX_NUM_WORDS, EMBEDDING_DIM, input_length=NUM_WORDS_SEQ))
model.add(Dropout(0.2))
model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, padding='same', activation='relu')) # filter_size - how many words to consider
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=25))

model.add(LSTM(128))
# model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='sigmoid'))
model.compile(loss=losses.sparse_categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
print(model.summary())



print('Training...')
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

df = load_data_csv(test_path)
if resample:

    df = resample_dataframe(df)


X_test, y_test = process_data(df, MAX_NUM_WORDS, NUM_WORDS_SEQ, labels=True)
acc = model.evaluate(X_test, y_test)

print('accuracy: {}'.format(acc))
with open('/logs/log.txt', 'r') as file:
    file.write('{} accuracy: {}\n'.format(str(datetime.datetime.now()), acc))


model.save(save_model_path)