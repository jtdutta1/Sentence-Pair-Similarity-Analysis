import numpy as np
import preprocess as pp
from keras.models import model_from_json

question1, question2 = pp.extract_data("quora-question-pairs/test.csv", 'test')
question1_word_sequences, question2_word_sequences, word_index = pp.tokenize(question1, question2)
embeddings_index = pp.get_embeddings("glove.840B.300d/glove.840B.300d.txt")
nb_words, word_embedding_matrix = pp.get_embedding_matrix(word_index, embeddings_index)
q1_data, q2_data, word_embedding_matrix, nb_words = pp.process_data(question1_word_sequences,
                                                                    question2_word_sequences,
                                                                    word_embedding_matrix,
                                                                    nb_words,
                                                                    'test')

X_train = np.stack((q1_data, q2_data), axis=1)
Q1_train = X_train[:, 0]
Q2_train = X_train[:, 1]

json_file = open('best_weights/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("best_weights/weights.h5")
print("Loaded model from disk")

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
score = model.predict([Q1_train, Q2_train])
print(score)
