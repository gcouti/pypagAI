'''
    Relational network for BABI, based on
    https://github.com/Alan-Lee123/relation-network

'''

from __future__ import print_function

import re
import tarfile
from functools import reduce

import keras.backend as K
import numpy as np
import theano.tensor as T
from keras.callbacks import LearningRateScheduler
from keras.layers import Input, merge
from keras.layers.core import Dense, Dropout, Lambda, Activation
from keras.layers.embeddings import Embedding
from keras.metrics import categorical_accuracy
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.preprocessing.sequence import pad_sequences


# np.random.seed(1337)  # for reproducibility
from keras.utils import get_file


class SequenceEmbedding(Embedding):
    def __init__(self, input_dim, output_dim, position_encoding=False, **kwargs):
        self.position_encoding = position_encoding
        self.zeros_vector =  T.zeros(output_dim, dtype='float32').reshape((1,output_dim))
        super(SequenceEmbedding, self).__init__(input_dim, output_dim, **kwargs)


    def call(self, x, mask=None):
        if 0. < self.dropout < 1.:
            retain_p = 1. - self.dropout
            B = K.random_binomial((self.input_dim,), p=retain_p) * (1. / retain_p)
            B = K.expand_dims(B)
            W = K.in_train_phase(self.W * B, self.W)
        else:
            W = self.W
        W_ = T.concatenate([self.zeros_vector, W], axis=0)
        out = K.gather(W_, x)
        return out

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q[:-1], a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append([sent[:-1]])
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data

def vectorize_facts(data, word_idx, story_maxlen, query_maxlen, fact_maxlen, enable_time = False):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = np.zeros((len(story), fact_maxlen),dtype='int32')
        for k,facts in enumerate(story):
            if not enable_time:
                x[k][-len(facts):] = np.array([word_idx[w] for w in facts])[:fact_maxlen]
            else:
                x[k][-len(facts)-1:-1] = np.array([word_idx[w] for w in facts])[:facts_maxlen-1]
                x[k][-1] = len(word_idx) + len(story) - k
        xq = [word_idx[w] for w in query]
        y = np.zeros(len(word_idx) + 1) if not enable_time else np.zeros(len(word_idx) + 1 + story_maxlen)
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return pad_sequences(X, maxlen=story_maxlen), pad_sequences(Xq, maxlen=query_maxlen), np.array(Y)

'''
try:
    path = get_file('babi-tasks-v1-2.tar.gz', origin='http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz')
except:
    print('Error downloading dataset, please download it manually:\n'
          '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
          '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
    raise
'''
path = get_file('babi-tasks-v1-2.tar.gz',  origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')

challenges = {
    # QA1 with 10,000 samples
    'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
    # QA2 with 10,000 samples
    'two_supporting_facts_10k': 'tasks_1-20_v1-2/en/qa2_two-supporting-facts_{}.txt',
    'three_supporting_facts_10k': 'tasks_1-20_v1-2/en/qa3_three-supporting-facts_{}.txt',

}
challenge_type = 'single_supporting_fact_10k'
challenge = challenges[challenge_type]

EMBED_HIDDEN_SIZE = 20
enable_time = True

with tarfile.open(path) as tar:
    print('Extracting stories for the challenge:', challenge_type)
    train_facts = get_stories(tar.extractfile(challenge.format('train')))
    test_facts = get_stories(tar.extractfile(challenge.format('test')))

train_stories = [(reduce(lambda x,y: x + y, map(list,fact)),q,a) for fact,q,a in train_facts]
test_stories = [(reduce(lambda x,y: x + y, map(list,fact)),q,a) for fact,q,a in test_facts]

facts_maxlen = max(map(len, (x for h,_,_ in train_facts + test_facts for x in h)))
if enable_time:
    facts_maxlen += 1

story_maxlen = max(map(len, (x for x, _, _ in train_facts + test_facts)))
query_maxlen = max(map(len, (x for _, x, _ in train_facts + test_facts)))

vocab = sorted(reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in train_stories + test_stories)))
# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
if enable_time:
    vocab_size += story_maxlen

print('-')
print('Vocab size:', vocab_size, 'unique words')
print('Story max length:', story_maxlen, 'words')
print('Query max length:', query_maxlen, 'words')
print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))
print('-')
print('Here\'s what a "story" tuple looks like (input, query, answer):')
print(train_stories[0])
print('-')
print('Vectorizing the word sequences...')

word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
inputs_train, queries_train, answers_train = vectorize_facts(train_facts, word_idx, story_maxlen, query_maxlen, facts_maxlen,
                                                             enable_time=enable_time)
inputs_test, queries_test, answers_test = vectorize_facts(test_facts, word_idx, story_maxlen, query_maxlen, facts_maxlen,
                                                          enable_time=enable_time)

print('-')
print('inputs: integer tensor of shape (samples, max_length)')
print('inputs_train shape:', inputs_train.shape)
print('inputs_test shape:', inputs_test.shape)
print('-')
print('queries: integer tensor of shape (samples, max_length)')
print('queries_train shape:', queries_train.shape)
print('queries_test shape:', queries_test.shape)
print('-')
print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
print('answers_train shape:', answers_train.shape)
print('answers_test shape:', answers_test.shape)
print('-')
print('Compiling...')


fact_input = Input(shape=(story_maxlen, facts_maxlen, ), dtype='int32', name='facts_input')
question_input = Input(shape=(query_maxlen, ), dtype='int32', name='query_input')

# input_length is different to input, so is not clear if I can share it, however this still
# work because embedding layer does not check that
question_layer = SequenceEmbedding(input_dim=vocab_size-1,
                                   output_dim=EMBED_HIDDEN_SIZE,
                                   input_length=query_maxlen, init='normal')

question_encoder = question_layer(question_input)
#question_encoder = Dropout(0.3)(question_encoder)
question_encoder = Lambda(lambda x: K.sum(x, axis=1),
                          output_shape=lambda shape: (shape[0],) + shape[2:])(question_encoder)


layer_encoder = SequenceEmbedding(input_dim=vocab_size-1,
                                  output_dim=EMBED_HIDDEN_SIZE,
                                  input_length=story_maxlen, init='normal')
input_encoder = layer_encoder(fact_input)

input_encoder = Lambda(lambda x: K.sum(x, axis=2),
                       output_shape=(story_maxlen, EMBED_HIDDEN_SIZE,))(input_encoder)
objects = []
for k in range(story_maxlen):
    fact_object = Lambda(lambda x: x[:,k,:], output_shape=(20,))(input_encoder)
    objects.append(fact_object)

relations = []
for fact_object_1 in objects:
    for fact_object_2 in objects:
        relations.append(merge([fact_object_1, fact_object_2, question_encoder], mode='concat',
                               output_shape=(None, EMBED_HIDDEN_SIZE * 3,)))

from keras.layers.normalization import BatchNormalization

MLP_unit = 64

def stack_layer(layers):
    def f(x):
        for k in range(len(layers)):
            x = layers[k](x)
        return x
    return f


def get_MLP(n):
    r = []
    for k in range(n):
        s = stack_layer([
            Dense(MLP_unit, input_shape=(EMBED_HIDDEN_SIZE * 3,)),
            BatchNormalization(),
            Activation('relu')
        ])
        r.append(s)
    return stack_layer(r)

g_MLP = get_MLP(3)
mid_relations = []
for r in relations:
    mid_relations.append(Dense(MLP_unit, input_shape=(EMBED_HIDDEN_SIZE,))(r))
combined_relation = merge(mid_relations, mode='sum')

def bn_dense(x):
    y = Dense(MLP_unit)(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Dropout(0.5)(y)
    return y

#rn = bn_dense(combined_relation)
response = Dense(vocab_size, init='uniform', activation='sigmoid')(combined_relation)

model = Model(input=[fact_input, question_input], output=[response])

#theano.printing.pydotprint(response, outfile="model.png", var_with_name_simple=True)
#plot(model, to_file='model.png')

def scheduler(epoch):
    if (epoch + 1) % 25 == 0:
        lr_val = model.optimizer.lr.get_value()
        model.optimizer.lr.set_value(lr_val*0.5)
    return float(model.optimizer.lr.get_value())

sgd = SGD(lr=0.01, clipnorm=40.)
adam = Adam(clipnorm = 40.)

print('Compiling model...')
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[categorical_accuracy])
print('Compilation done...')


lr_schedule = LearningRateScheduler(scheduler)

model.fit([inputs_train, queries_train], answers_train,
          batch_size=32,
          nb_epoch=100,
          validation_split=0.1,
          callbacks=[lr_schedule],
          verbose=1)

loss, acc = model.evaluate([inputs_test, queries_test], answers_test)

print(loss,acc)