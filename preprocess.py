import json
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def extract_data(filepath, dtype):
    """
    Extracts the training/testing data from the .csv provided.
    Dataset download link: https://www.kaggle.com/c/quora-question-pairs/data

    :param filepath: System path to the data file
    :param dtype: Type of data to extract. eg: 'train', 'test', etc.
    :return: Returns feature (and labels for dtype = 'training') data as a list
    """
    data = pd.read_csv(filepath).dropna()
    question1 = data['question1'].tolist()
    question2 = data['question2'].tolist()
    if dtype == 'train':
        is_duplicate = data['is_duplicate'].tolist()
        print("Extraction complete")
        return question1, question2, is_duplicate
    else:
        print("Extraction complete")
        return question1, question2


def tokenize(question1, question2):
    """
    Tokenize the sentences into word sequences eliminating spaces and associated punctuations.
    Attribute num_words = 200k has been adopted for best results over 50k, 100k, 200k and 300k.
    num_words: the maximum number of words to keep, based on word frequency.

    By default, all punctuation is removed, turning the texts into space-separated sequences of words.
    These sequences are then split into lists of tokens. They will then be indexed or vectorized.
    (0 is a reserved index that won't be assigned to any word.)

    :param question1: Data from the first sentence
    :param question2: Data from the second sentence
    :return: Returns the word sequences for the two sentences along with the combined word indexes
    """
    questions = question1 + question2
    tokenizer = Tokenizer(num_words=200000)
    tokenizer.fit_on_texts(questions)
    question1_word_sequences = tokenizer.texts_to_sequences(question1)
    question2_word_sequences = tokenizer.texts_to_sequences(question2)
    word_index = tokenizer.word_index
    print("Tokenizing complete")
    return question1_word_sequences, question2_word_sequences, word_index


def get_embeddings(filepath):
    """
    Process values from pre-trained GloVe embeddings.
    GloVe is an unsupervised learning algorithm for obtaining vector representations for words.
    Pre-trained word vectors. Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download):
    Download link: http://nlp.stanford.edu/data/glove.840B.300d.zip

    :param filepath: System path to the data file
    :return: Returns an embedding dictionary for the type of vocabulary provided, i.e. 400k, 1.9m, 2.2m, etc.
    """
    embeddings_index = {}
    with open(filepath, encoding='utf-8') as file:
        for line in file:
            values = line.split(' ')
            gword = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings_index[gword] = embedding
    print("Embeddings complete")
    return embeddings_index


def get_embedding_matrix(word_index, embeddings_index):
    """
    Prepares the word embedding matrix, encoding data from GloVe dictionary as was provided in get_embeddings()

    :param word_index: Combined word indexes as returned after tokenizing from both sentences
    :param embeddings_index: Embedding dictionary as returned from the pre-trained GloVe word vectors
    :return: Returns the minimum of length of the most common words from the embedding dictionary and the word indexes
        Also returns a matrix containing embeddings prepared from the pre-trained embedding dictionary for the sentences
    """
    nb_words = min(200000, len(word_index))
    word_embedding_matrix = np.zeros((nb_words + 1, 300))
    for word, i in word_index.items():
        if i > 200000:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            word_embedding_matrix[i] = embedding_vector
    print("Embedding matrix complete")
    return nb_words, word_embedding_matrix


def process_data(question1_word_sequences, question2_word_sequences, word_embedding_matrix, nb_words, dtype, is_duplicate=None):
    """
    The og function that does it all.

    :param question1_word_sequences: As returned from tokenize() for first sentence
    :param question2_word_sequences: As returned from tokenize() for second sentence
    :param word_embedding_matrix: As returned from get_embedding_matrix()
    :param nb_words: As returned from get_embedding_matrix()
    :param dtype: Type of data to process. eg: 'train', 'test', etc.
    :param is_duplicate: As returned from extract_data() (Only applicable for dtype = 'train)
    :return: Returns processed word vectors for both sentences (and labels for dtype = 'train'
        Returns the minimum of length of the most common words from the embedding dictionary and the word indexes
        Also returns a matrix containing embeddings prepared from the pre-trained embedding dictionary for the sentences
    """
    q1_data = pad_sequences(question1_word_sequences, maxlen=25)
    q2_data = pad_sequences(question2_word_sequences, maxlen=25)
    if dtype == 'train':
        labels = np.array(is_duplicate, dtype=int)
        print("Processing complete")
        return q1_data, q2_data, labels, word_embedding_matrix, nb_words
    else:
        print("Processing complete")
        return q1_data, q2_data, word_embedding_matrix, nb_words
