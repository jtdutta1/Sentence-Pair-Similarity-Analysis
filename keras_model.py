from keras.models import *
from keras.layers import *
from keras.layers.embeddings import *
from keras.utils.vis_utils import plot_model
import numpy as np


def model(nb_words, word_embedding_matrix):
    """
    Defines the Siamese-Dense RNN (Recurrent Neural Network.
    Input shape for first sentence = (None, 25)
    Input shape for second sentence = (None, 25)

    :param nb_words: Combined word indexes as returned after tokenizing from both sentences
    :param word_embedding_matrix: Embedding dictionary as returned from the pre-trained GloVe word vectors
    :return: Returns the compiled model instance
    """
    input_1 = Input(shape=(25,))
    input_2 = Input(shape=(25,))

    q1 = Embedding(input_dim=nb_words + 1,
                   output_dim=300,
                   weights=[word_embedding_matrix],
                   input_length=25,
                   trainable=False)(input_1)
    q1 = TimeDistributed(Dense(300, activation='relu'))(q1)
    q1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(300,))(q1)

    q2 = Embedding(input_dim=nb_words + 1,
                   output_dim=300,
                   weights=[word_embedding_matrix],
                   input_length=25,
                   trainable=False)(input_2)
    q2 = TimeDistributed(Dense(300, activation='relu'))(q2)
    q2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(300,))(q2)

    x = concatenate([q1, q2])
    x = Dense(200, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)
    x = Dense(200, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)
    x = Dense(200, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)
    x = Dense(200, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)

    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input_1, input_2], outputs=out)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
