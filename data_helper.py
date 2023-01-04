import numpy as np
import pandas as pd
import codecs
import pickle
import os
from keras_bert import Tokenizer
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from keras_bert import get_custom_objects
from tensorflow.keras.models import model_from_json
from datetime import datetime
from tensorflow.keras import optimizers


def data_processor(DATA, INPUTS_REP, MODEL_TYPE, BERT_VER, MAX_SEQUENCE_LENGTH, VALIDATION_SPLIT, OUTPUT_PATH):
    print('Load {} dataset:'.format(DATA))
    CONFIG_FILE, CHECKPOINT_FILE, DICT_FILE, EMBED_DIM = load_bert(BERT_VER)
    inputs, counts_text, counts_tc, counts_time, counts_label = load_data(DATA, INPUTS_REP, MODEL_TYPE)

    text_encoded = []
    text_encoded_file = f'{OUTPUT_PATH}/{DATA}-{INPUTS_REP[0]}-{INPUTS_REP[1]}.p'

    if os.path.exists(text_encoded_file):
        print('Load encoded text:')
        text_encoded = pickle.load(open(text_encoded_file, 'rb'))
    else:
        token_dict = {}
        with codecs.open(DICT_FILE, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)
        tokenizer = Tokenizer(token_dict)
        print('Process text:')
        for feature in inputs[:counts_text]: text_encoded.extend(data_encoder(tokenizer, feature, MAX_SEQUENCE_LENGTH))
        pickle.dump(text_encoded, open(text_encoded_file, 'wb'))

    tc_encoded, counts_unique_tc = [], []
    if counts_tc:
        print('Process categorical text inputs:')
        tc_encoded = [seq_padding_category(to_categorical(f), EMBED_DIM) for f in
                      inputs[counts_text:counts_text + counts_tc]]
        counts_unique_tc = [len(set(f)) for f in inputs[counts_text:counts_text + counts_tc]]

    time_encoded, counts_unique_time = [], []
    if counts_time:
        print('Process time inputs:')
        if INPUTS_REP[1] == 'uniform':
            time_encoded = [np.asarray(f) for f in inputs[-(counts_time + counts_label):-counts_label]]
            counts_unique_time = [len(set(f)) for f in inputs[-(counts_time + counts_label):-counts_label]]
        else:
            time_encoded = [seq_padding_category(to_categorical(f), EMBED_DIM) for f in
                            inputs[-(counts_time + counts_label):-counts_label]]
            counts_unique_time = [len(set(f)) for f in inputs[-(counts_time + counts_label):-counts_label]]

    print('Process labels: ')
    labels, number_classes = [], []
    for i in range(counts_label):
        labels.append(np.array(inputs[-i - 1]))
        number_classes.append(np.max(labels[-1]) + 1)

    print('Shuffle data: ')
    np.random.seed(12345)
    indices = np.arange(len(text_encoded[0]))
    np.random.shuffle(indices)

    text_encoded = [x[indices] for x in text_encoded]
    tc_encoded = [x[indices] for x in tc_encoded]
    time_encoded = [x[indices] for x in time_encoded]
    labels = [x[indices] for x in labels]

    print('Split data: ')
    nvs = int(VALIDATION_SPLIT * len(text_encoded[0]))  # num_validation_samples

    train_text, val_text, test_text = [f[:-2 * nvs] for f in text_encoded], [f[-2 * nvs:-nvs] for f in text_encoded], [
        f[-nvs:] for f in text_encoded]
    train_tc, val_tc, test_tc = [f[:-2 * nvs] for f in tc_encoded], [f[-2 * nvs:-nvs] for f in tc_encoded], [f[-nvs:]
                                                                                                             for f in
                                                                                                             tc_encoded]
    train_time, val_time, test_time = [f[:-2 * nvs] for f in time_encoded], [f[-2 * nvs:-nvs] for f in time_encoded], [
        f[-nvs:] for f in time_encoded]
    train_labels, val_labels, test_labels = [f[:-2 * nvs] for f in labels], [f[-2 * nvs:-nvs] for f in labels], [
        f[-nvs:] for f in labels]

    correla_matircs = getCorrelaMatrics(number_classes) if MODEL_TYPE == 'mtlBERT' else []

    return [train_text, val_text, test_text, train_tc, val_tc, test_tc, train_time, val_time, test_time, train_labels,
            val_labels, test_labels, number_classes, counts_text, counts_unique_tc, counts_unique_time, correla_matircs]


def load_data(DATA, INPUTS_REP, MODEL_TYPE, test=False):
    # inputs order: text, tc (textual categories), time, label
    files = {
            'Flickr': 'datasets/flickr-mel-posts-ex.csv',
             'Twitter-Mel': 'datasets/twitter-mel-posts-ex.csv'}
    DATA_FILE = files[DATA]
    dfData = pd.read_csv(DATA_FILE)
    dfData = dfData.fillna('Nil')
    if test: dfData = dfData.sample(n=100, random_state=1)

    label_L3, counts_label = dfData['poiID_unix'].tolist() if DATA == 'Twitter-SM' else dfData['poiID'].tolist(), 1

    le = preprocessing.LabelEncoder()
    counts_tc, counts_time = 0, 0

    if DATA == 'Flickr':
        title = [line.strip() for line in dfData['title']]
        description = [line.strip() for line in dfData['description']]
        userName = dfData['userName'].tolist()
        userTag = dfData['userTags'].tolist()
        machineTag = dfData['machineTags'].tolist()
        userDesp = dfData['userDesp'].tolist()
        occupation = dfData['occupation'].tolist()
        hometown = dfData['hometown'].tolist()
        city = dfData['city'].tolist()
        country = dfData['country'].tolist()
        if INPUTS_REP[1] == 'text':
            takeAt = dfData['takenAt'].tolist()
            postAt = dfData['postedAt'].tolist()
            joinAt = dfData['joinedAt'].tolist()
            inputs = [title, description, userName, userTag, machineTag, userDesp, takeAt, postAt, joinAt,
                      occupation, hometown, city, country, label_L3]
        else:
            takenHour = dfData['takenAtHour'].tolist()
            takenWeekday = dfData['takenAtWeekday'].tolist()
            takenMonth = dfData['takenAtMonth'].tolist()
            uploadedHour = dfData['postedAtHour'].tolist()
            uploadedWeekday = dfData['postedAtWeekday'].tolist()
            uploadedMonth = dfData['postedAtMonth'].tolist()
            joinHour = dfData['joinedAtHour'].tolist()
            joinWeekday = dfData['joinedAtWeekday'].tolist()
            joinMonth = dfData['joinedAtMonth'].tolist()
            counts_time = 9
            inputs = [title, description, userName, userTag, machineTag, userDesp, occupation, hometown, city, country,
                      takenHour, takenWeekday, takenMonth, uploadedHour, uploadedWeekday, uploadedMonth,
                      joinHour, joinWeekday, joinMonth, label_L3]
    elif DATA == 'Twitter-Mel':
        texts = [line.strip() for line in dfData['text']]
        userDesp = list(dfData['userDesp'])
        userLoc = list(dfData['userLoc'])

        if INPUTS_REP[0] == 'text':
            source = list(dfData['source'])
            media = list(dfData['media'])
        else:
            source = le.fit_transform(dfData['source_128'].tolist())
            media = le.fit_transform(dfData['media'].tolist())
            counts_tc = 2
        if INPUTS_REP[1] == 'text':
            created_at = list(dfData['created_at'])
            inputs = [texts, userDesp, userLoc, created_at, source, media, label_L3]
        else:
            createdHours = list(dfData['createdHour'])
            createdWeekdays = list(dfData['createdWeekday'])
            createdMonths = list(dfData['createdMonth'])
            inputs = [texts, userDesp, userLoc, source, media, createdHours, createdWeekdays, createdMonths, label_L3]
            counts_time = 3
    counts_text = len(inputs) - (counts_tc + counts_time + counts_label)

    if MODEL_TYPE in ['mtlTagger', 'hierTagger'] :
        inputs.extend([dfData['subThemeID'].tolist(), dfData['themeID'].tolist()])
        counts_label += 2
    return inputs, counts_text, counts_tc, counts_time, counts_label


def seq_padding(X, padding=0, ML=0):
    ML = max([len(x) for x in X]) if ML == 0 else ML
    return np.array([np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x[:ML] for x in X])


def seq_padding_category(X, EMBEDDING_DIM, padding=0):
    '''
    pad sentence to MAX_SEQUENCE_LENGTH with 0 for those len<MAX_SEQUENCE_LENGTH
    '''
    ML = EMBEDDING_DIM
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


def load_bert(bert_version, bert_path='BERT/'):
    files = ['uncased_L-2_H-128_A-2', 'uncased_L-4_H-256_A-4', 'uncased_L-4_H-512_A-8', 'uncased_L-8_H-512_A-8',
             'uncased_L-12_H-768_A-12']
    bert_info = dict({'Tiny': [bert_path + files[0] + '/bert_config.json',
                               bert_path + files[0] + '/bert_model.ckpt',
                               bert_path + files[0] + '/vocab.txt',
                               128],
                      'Mini': [bert_path + files[1] + '/bert_config.json',
                               bert_path + files[1] + '/bert_model.ckpt',
                               bert_path + files[1] + '/vocab.txt',
                               256],
                      'Small': [bert_path + files[2] + '/bert_config.json',
                                bert_path + files[2] + '/bert_model.ckpt',
                                bert_path + files[2] + '/vocab.txt',
                                512],
                      'Medium': [bert_path + files[3] + '/bert_config.json',
                                 bert_path + files[3] + '/bert_model.ckpt',
                                 bert_path + files[3] + '/vocab.txt',
                                 512],
                      'Base': [bert_path + files[4] + '/bert_config.json',
                               bert_path + files[4] + '/bert_model.ckpt',
                               bert_path + files[4] + '/vocab.txt',
                               768]
                      })
    return bert_info[bert_version]


def data_encoder(tokenizer, feature, MAX_SEQUENCE_LENGTH=0):
    tokens = [tokenizer.encode(first=x) for x in feature]
    indexWord, indexSeq = [x[0] for x in tokens], [x[1] for x in tokens]
    return [seq_padding(indexWord, ML=MAX_SEQUENCE_LENGTH), seq_padding(indexSeq, ML=MAX_SEQUENCE_LENGTH)]


def getCorrelaMatrics(number_classes):
    dfPOI = pd.read_csv('datasets/poi-mel-ex.csv')
    CORRELA_MATRIX_1to2 = calCorrelaMatrix(dfPOI, number_classes[0], number_classes[1], 'themeID', 'subThemeID')
    CORRELA_MATRIX_1to3 = calCorrelaMatrix(dfPOI, number_classes[0], number_classes[2], 'themeID', 'poiID')
    CORRELA_MATRIX_2to3 = calCorrelaMatrix(dfPOI, number_classes[1], number_classes[2], 'subThemeID', 'poiID')
    return [CORRELA_MATRIX_1to2, CORRELA_MATRIX_1to3, CORRELA_MATRIX_2to3]


def calCorrelaMatrix(dfPOI, label1, label2, label1_name='themeID', label2_name='poiID'):
    label_L1_COL = label1_name
    label_L2_COL = label2_name
    nrows = dfPOI[label_L1_COL].nunique()
    ncols = dfPOI[label_L2_COL].nunique()
    CORRELA_MATRIX = np.zeros([nrows, ncols]) - 1
    for i in range(nrows):
        temp = dfPOI.loc[(dfPOI[label_L1_COL] == i)][label_L2_COL].tolist()
        for j in temp:
            CORRELA_MATRIX[i - 1][j - 1] = 0
    CORRELA_MATRIX = CORRELA_MATRIX[:label1, :label2]
    CORRELA_MATRIX = concatenate(
        [CORRELA_MATRIX[:round(CORRELA_MATRIX.shape[0] / 2)], CORRELA_MATRIX[round(CORRELA_MATRIX.shape[0] / 2):]],
        axis=0)
    return CORRELA_MATRIX


def data_split(train_text, train_tc, train_time, val_text, val_tc, val_time, train_labels, val_labels, number_classes):
    train_text_L2 = [[np.asarray([x[i] for i in range(len(x)) if train_labels[0][i] == j]) for x in train_text] for j in
                     range(number_classes[0])]
    train_tc_L2 = [[np.asarray([x[i] for i in range(len(x)) if train_labels[0][i] == j]) for x in train_tc] for j in
                   range(number_classes[0])]
    train_time_L2 = [[np.asarray([x[i] for i in range(len(x)) if train_labels[0][i] == j]) for x in train_time] for j in
                     range(number_classes[0])]

    val_text_L2 = [[np.asarray([x[i] for i in range(len(x)) if val_labels[0][i] == j]) for x in val_text] for j in
                   range(number_classes[0])]
    val_time_L2 = [[np.asarray([x[i] for i in range(len(x)) if val_labels[0][i] == j]) for x in val_time] for j in
                   range(number_classes[0])]
    val_tc_L2 = [[np.asarray([x[i] for i in range(len(x)) if val_labels[0][i] == j]) for x in val_tc] for j in
                 range(number_classes[0])]
    train_labels_L2 = [
        np.array([train_labels[-1][j] for j in range(train_labels[-1].shape[0]) if train_labels[0][j] == i]) for i in
        range(number_classes[0])]
    val_labels_L2 = [np.array([val_labels[-1][j] for j in range(val_labels[-1].shape[0]) if val_labels[0][j] == i]) for i
                     in range(number_classes[0])]
    number_classes.append(np.array([np.max(x) + 1 if x.shape[0] > 0 else 0 for x in train_labels_L2]))

    return train_text_L2, train_tc_L2, train_time_L2, val_text_L2, val_time_L2, val_tc_L2, train_labels_L2, val_labels_L2, number_classes


def load_model_jh(MODEL_NAME, LOSS, LEARNING_RATE, note):
    print('Load model: {}'.format(note))
    json_file = open(MODEL_NAME + '.json', 'r')
    model = model_from_json(json_file.read(), custom_objects=get_custom_objects())
    json_file.close()
    model.load_weights(MODEL_NAME + '.h5')
    model.compile(loss=LOSS, optimizer=optimizers.Adam(learning_rate=LEARNING_RATE), metrics=['acc'])
    return model


def save_model_jh(model, MODEL_NAME):
    with open(MODEL_NAME + '.json', "w") as json_file:
        json_file.write(model.to_json())
    model.save_weights(MODEL_NAME + '.h5')


def log_history(LOG_FILE, OUTPUT_FILE, anno, history, note=None, muted=False):
    # muted: False: show anno (default), True: not show anno
    with open(LOG_FILE, 'a') as f:
        if not muted:
            f.write('{} output of {}.py:\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), OUTPUT_FILE))
            f.write(anno + '\n')
        if note: f.write(note + '\n')
        for key, value in history.history.items():
            f.write(key + ': ')
            for i in value: f.write('{:.4}, '.format(i))
        f.write('\n')
