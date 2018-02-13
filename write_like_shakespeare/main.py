from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from shakespeare_utils import on_epoch_end, build_data, vectorization, generate_output
import sys
import io


def main():
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

    print("Loading text data...")
    text = io.open('shakespeare.txt', encoding='utf-8').read().lower()
    # print('corpus length:', len(text))

    Tx = 40
    chars = sorted(list(set(text)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    # print('number of unique characters in the corpus:', len(chars))

    print("Creating training set...")
    X, Y = build_data(text, Tx, stride=3)
    print("Vectorizing training set...")
    x, y = vectorization(X, Y, n_x=len(chars), char_indices=char_indices)
    print("Loading model...")
    model = load_model('models/model_shakespeare_kiank_350_epoch.h5')

    # model.fit(x, y, batch_size=128, epochs=1, callbacks=[print_callback])
    model.fit(x, y, batch_size=128, epochs=1)

    # Run this cell to try with different inputs without having to re-train the model
    generate_output(model, chars, char_indices, indices_char, Tx)

if __name__ == '__main__':
    main()