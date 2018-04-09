import numpy as np
import preprocess as pp
from keras.models import *
from keras_model import model

question1, question2, is_duplicate = pp.extract_data("quora-question-pairs/train.csv", 'train')
question1_word_sequences, question2_word_sequences, word_index = pp.tokenize(question1, question2)
embeddings_index = pp.get_embeddings("glove.840B.300d/glove.840B.300d.txt")
nb_words, word_embedding_matrix = pp.get_embedding_matrix(word_index, embeddings_index)
q1_data, q2_data, labels, word_embedding_matrix, nb_words = pp.process_data(question1_word_sequences,
                                                                            question2_word_sequences,
                                                                            word_embedding_matrix,
                                                                            nb_words,
                                                                            'train',
                                                                            is_duplicate)

X_train = np.stack((q1_data, q2_data), axis=1)
y_train = labels
Q1_train = X_train[:, 0]
Q2_train = X_train[:, 1]

model = model(nb_words, word_embedding_matrix)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit([Q1_train, Q2_train],
          y_train,
          batch_size=32,
          epochs=100)

model_json = model.to_json()
with open("best_weights/model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("best_weights/weights.h5")
print("Saved model to disk")
