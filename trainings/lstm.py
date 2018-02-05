import sys
import datetime
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from data_processing  import process_data, resample_dataframe, load_data_csv


#ARGS: train path test path modelsave path

cmd_arg = sys.argv
assert len(cmd_arg) == 4
train_path = cmd_arg[1]
test_path = cmd_arg[2]
save_model_path = cmd_arg[3]

MAX_NUM_WORDS = 50000
NUM_WORDS_SEQ = 100
EMBEDDING_DIM = 300

df = load_data_csv(train_path)
resampled_df = resample_dataframe(df)

X_train, y_train = process_data(resampled_df, MAX_NUM_WORDS, NUM_WORDS_SEQ, labels=True)

# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)



print('Initializing model...')
model = Sequential()
model.add(Embedding(MAX_NUM_WORDS, EMBEDDING_DIM, input_length=NUM_WORDS_SEQ))
model.add(Dropout(0.2))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')) # filter_size - how many words to consider
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


print('Training...')
model.fit(X_train, y_train, batch_size=100, epochs=10)

df = load_data_csv(test_path)

resampled_df = resample_dataframe(df)

X_test, y_test = process_data(resampled_df, MAX_NUM_WORDS, NUM_WORDS_SEQ, labels=True)
acc = model.evaluate(X_test, y_test)

print('accuracy: {}'.format(acc))
with open('../logs/log.txt', 'r') as file:
    file.write('{} accuracy: {}\n'.format(str(datetime.datetime.now()), acc))


model.save(save_model_path)
# pickle.dump(model, save_model_path)