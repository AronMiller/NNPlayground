'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
from collections import Counter
from keras.models import model_from_json

output_folder = './model1/'


def buildModel():
	# build the model: 2 stacked LSTM
	print('Build model...')
	model = Sequential()

	# model.add(Dense(512, input_dim= maxlen * len(chars), init='uniform'))
	# model.add(Activation('relu'))
	# model.add(Dropout(0.1))
	'''
	model.add(Dense(256, input_dim= maxlen * len(chars), init='uniform'))
	model.add(Activation('relu'))
	model.add(Dropout(0.1))

	model.add(Dense(128, input_dim= maxlen * len(chars), init='uniform'))
	model.add(Activation('relu'))
	model.add(Dropout(0.1))

	model.add(Dense(64, input_dim= maxlen * len(chars), init='uniform'))
	model.add(Activation('relu'))
	model.add(Dropout(0.1))
	'''

	# model.add(LSTM(9, return_sequences=False, input_shape=(maxlen, len(chars))))
	model.add(LSTM(32, return_sequences=True, input_shape=(maxlen, len(chars))))
	model.add(Dropout(0.1))

	model.add(LSTM(32, return_sequences=False))
	model.add(Dropout(0.1))

	model.add(Dense(32, init='uniform'))
	model.add(Activation('relu'))
	model.add(Dropout(0.1))

	#model.add(LSTM(32, return_sequences=True))
	#model.add(Dropout(0.1))
	#model.add(LSTM(32, return_sequences=False))
	#model.add(Dropout(.1))
	model.add(Dense(len(chars)))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


	# Save model
	json_string = model.to_json()
	model_file = open(output_folder + "model.json", "w+")
	model_file.write(json_string)
	model_file.close()

	return model

# path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")

try: 
    text = open('/home/aron/Code/TextLSTM/holmes.txt').read().lower() #.translate(str.maketrans('','','\r\n'))

except UnicodeDecodeError:
    import codecs
    text = codecs.open(path, encoding='utf-8').read().lower()


chars = set(text)
print('total chars:', len(chars))
print('corpus length:', len(text))

# Remove less used characters
'''
x = Counter(text)
text = text.translate(str.maketrans('','', ''.join([z for z in x if x[z] < .001 * len(text)])))  

chars = set(text)
print('total chars after trim:', len(chars))
print('corpus length:', len(text))
'''

# Remove excessive spaces
'''
textSplit = text.split(' ')
textSplit2 = [z for z in textSplit if z != '']
text = ' '.join(textSplit2)
'''

print(text[:4000])
print('corpus length:', len(text))

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 32
step = 100

sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
# X = np.zeros((len(sentences), maxlen * len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
        # X[i, t * len(chars) + char_indices[char]]= 1
    y[i, char_indices[next_chars[i]]] = 1





def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

def completePhrase(sentence, text, model):
        generated = ''
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(4000):
            x = np.zeros((1, maxlen, len(chars)))
            #x = np.zeros((1, maxlen * len(chars)))
            for t, char in enumerate(sentence):
                #x[0, t * len(chars) + char_indices[char]] = 1
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()

# model = buildModel()
model = model_from_json(open(output_folder + 'model.json').read())
iter_start = 1
model.load_weights(output_folder + 'weights' + str(iter_start).rjust(5, '0'))

# train the model, output generated text after each iteration
for iteration in range(iter_start + 1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=100, nb_epoch=1)
    model.save_weights(output_folder + 'weights' + str(iteration).rjust(5, '0'))

    start_index = random.randint(0, len(text) - maxlen - 1)
    sentence = text[start_index: start_index + maxlen]

    for diversity in [0.25, 0.5, 0.75, 1.0]:
        print()
        print('----- diversity:', diversity)
        completePhrase(sentence, text, model)
        print()

    userCommand = ''
    while (True):
        userCommand = input('Enter prompt (Enter to continue):')
        if(userCommand == ''):
            break
        userCommand = userCommand.rjust(maxlen, ' ')
        userCommand = userCommand[-maxlen:]
        diversity=0.2
        completePhrase(userCommand, text, model)
        print()
