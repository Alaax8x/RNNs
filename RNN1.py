import numpy as np
import os
from pickle import dump
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, GRU, Embedding, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping


# Cleaning the data (Assumes pre-cleaned data in 'data/mediumcleaned.txt')
dataset_path = "data/mediumcleaned.txt"
data = open(dataset_path, 'rb').read().decode(encoding='utf-8')

# Splitting the data into sentences
sentences = data.splitlines()

# Creating tokenizer
def create_tokenizer(sentences):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    return tokenizer

tokenizer = create_tokenizer(sentences)
dump(tokenizer, open('tokenizer.p', 'wb'))
vocab_size = len(tokenizer.word_index) + 1 # number of unique words 2922

lengths = [len(sentence.split()) for sentence in sentences]

print(f"Sentence lengths: {min(lengths)}")
print(f"Sentence lengths: {max(lengths)}")
print(f"Mean length: {np.mean(lengths)}")

# Calculating maximum length of sentences
def max_length(sentences_with_tokens, max_threshold=700):
    return min(max(len(d.split()) for d in sentences_with_tokens), max_threshold)

max_length = max_length(sentences, max_threshold=700)


# Group sentences into a dictionary
def group_sentences_into_lists(sentences, group_size):
    grouped_lists = []
    for i in range(0, len(sentences), group_size):
        grouped_lists.append(sentences[i:i+group_size])
    return grouped_lists

def convert_to_dictionary(sentences, group_size):
    grouped_sentences = group_sentences_into_lists(sentences, group_size)
    dictionary = {}
    for i, group in enumerate(grouped_sentences, start=1):
        dictionary[i] = group
    return dictionary

group_size = 5
sentences_dictionary = convert_to_dictionary(sentences, group_size)

# Data generator with batching
def data_generator(descriptions, tokenizer, max_length, batch_size=128):
    X1, y = [], []
    while True:
        for key, description_list in descriptions.items():
            input_sequence, output_word = create_sequences(tokenizer, max_length, description_list)
            X1.extend(input_sequence)
            y.extend(output_word)
            if len(X1) >= batch_size:
                yield (np.array(X1[:batch_size]), np.array(y[:batch_size]))
                X1, y = X1[batch_size:], y[batch_size:]

def create_sequences(tokenizer, max_length, desc_list):
    X1, y = list(), list()
    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(y)

# Validation generator
def validation_generator(descriptions, tokenizer, max_length, batch_size=128):
    X1, y = [], []
    while True:
        for key, description_list in descriptions.items():
            input_sequence, output_word = create_sequences(tokenizer, max_length, description_list)
            X1.extend(input_sequence)
            y.extend(output_word)
            if len(X1) >= batch_size:
                yield (np.array(X1[:batch_size]), np.array(y[:batch_size]))
                X1, y = X1[batch_size:], y[batch_size:]


# Define the model
def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(max_length,), name="input_layer")
    se1 = Embedding(vocab_size, 128, mask_zero=True, name="embedding")(inputs1)
    se2 = Dropout(0.3, name="dropout1")(se1)
    se3 = GRU(512, return_sequences=True, name="gru1")(se2)
    se4 = Dropout(0.3, name="dropout2")(se3)
    se5 = GRU(512, name="gru2")(se4)
    decoder1 = Dense(256, activation='relu', name="dense1")(se5)
    dropout1 = Dropout(0.3, name="dropout3")(decoder1)
    decoder2 = Dense(256, activation='relu', name="dense2")(dropout1)
    outputs = Dense(vocab_size, activation='softmax', name="output_layer")(decoder2)

    model = Model(inputs=[inputs1], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    print(model.summary())
    return model

# Split sentences into training and validation sets
train_sentences, val_sentences = train_test_split(sentences, test_size=0.2, random_state=42)
train_sentences_dict = convert_to_dictionary(train_sentences, group_size)
val_sentences_dict = convert_to_dictionary(val_sentences, group_size)

# Define the model
model = define_model(vocab_size, max_length)

epochs = 20
batch_size = 32
steps = max(1, len(train_sentences_dict) // batch_size)
val_steps = max(1, len(val_sentences_dict) // batch_size)

if not os.path.exists("models"):
    os.mkdir("models")

# Add Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Training with validation
train_generator = data_generator(train_sentences_dict, tokenizer, max_length, batch_size)
val_generator = validation_generator(val_sentences_dict, tokenizer, max_length, batch_size)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    steps_per_epoch=steps,
    validation_steps=val_steps,
    verbose=1,
    callbacks=[early_stopping]
)

# Save the final model
model.save("predict_RRN.keras")

final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print(f"Final Training Loss: {final_train_loss}")
print(f"Final Validation Loss: {final_val_loss}")
