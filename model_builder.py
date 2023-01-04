from keras_bert import load_trained_model_from_checkpoint
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from keras_transformer import get_encoders
from tensorflow.keras import optimizers
from keras_pos_embd import TrigPosEmbedding
from data_helper import load_bert
import tensorflow as tf
from keras_bert import get_custom_objects
from tensorflow.keras.models import model_from_json
from datetime import datetime
import numpy as np
from math import radians, cos, sin, asin, sqrt, log
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd


def build_transTagger(INPUTS_REP, MODEL_TYPE, BERT_VER, BERT_TRAINABLE, POSITION, ENCODER, LEARNING_RATE, LOSS,
                    number_classes, counts_text, counts_unique_tc, counts_unique_time, correla_matrics, muted=False):
    # muted: to show model summary or not, default False (show)
    inputs = []
    config_file, checkpoint_file, dict_file, embed_dim = load_bert(BERT_VER)
    bert_model = load_trained_model_from_checkpoint(config_file, checkpoint_file, seq_len=None)
    for l in bert_model.layers: l.trainable = BERT_TRAINABLE

    alpha_bert = []
    for i in range(counts_text):
        input_indexWord = Input(shape=(None,), name='input_indexword_' + str(i))
        input_indexSeq = Input(shape=(None,), name='input_indexseq_' + str(i))
        tempLayer = bert_model([input_indexWord, input_indexSeq])
        tempLayer = Lambda(lambda x: x[:, 0])(tempLayer)
        alpha_bert.append(tempLayer)
        inputs.extend([input_indexWord, input_indexSeq])

    alpha_tc = []
    if INPUTS_REP[0] == '1hot':
        for i, c in enumerate(counts_unique_tc):
            input_tc = Input(shape=(embed_dim,), name='input_tc_' + str(i))
            alpha_tc.append(input_tc)
            inputs.append(input_tc)

    alpha_time = []
    if INPUTS_REP[1] == 'uniform':
        embed_time = []
        for i, c in enumerate(counts_unique_time):
            input_time = Input(shape=(1,), name='input_time_' + str(i))
            embed_temp = Embedding(c, embed_dim, trainable=True)(input_time)
            embed_time.append(embed_temp)
            inputs.append(input_time)
        alpha_time.append(Flatten()(sum(embed_time)))
    elif INPUTS_REP[1] == '1hot':
        for i, c in enumerate(counts_unique_time):
            input_time = Input(shape=(embed_dim,), name='input_time_' + str(i))
            alpha_time.append(input_time)
            inputs.append(input_time)

    alpha = concatenate(alpha_bert + alpha_tc + alpha_time, axis=1)

    if MODEL_TYPE == 'Tagger':
        preds = Dense(number_classes[-1], activation='softmax')(alpha)
        model = Model(inputs, preds)
        model.compile(loss=LOSS, optimizer=optimizers.Adam(learning_rate=LEARNING_RATE), metrics=['acc'])

    else:
        alpha = Reshape((len(alpha_bert + alpha_tc + alpha_time), embed_dim))(alpha)
        if POSITION == 'con':
            alpha = TrigPosEmbedding(mode=TrigPosEmbedding.MODE_CONCAT, output_dim=embed_dim, name='position-embed')(
                alpha)
        else:
            alpha = TrigPosEmbedding(mode=TrigPosEmbedding.MODE_ADD, name='position-embed')(alpha)
        alpha = get_encoders(input_layer=alpha, encoder_num=ENCODER[0], head_num=ENCODER[1], hidden_dim=ENCODER[2],
                             dropout_rate=ENCODER[3])
        alpha = Flatten()(alpha)
        if MODEL_TYPE == 'mtlTagger':
            pred1 = Dense(number_classes[0], activation='softmax', name='L1_output')(alpha)
            pred2 = Dense(number_classes[1], activation='softmax', name='L2_output')(alpha)
            # bias_1to3, bias_2to3 = tf.matmul(pred1, correla_matrics[1]), tf.matmul(pred2, correla_matrics[2])
            # bias = Add()([bias_1to3, bias_2to3])
            # beta = concatenate([alpha, bias], name='beta')
            # beta = concatenate([alpha, bias_1to3, bias_2to3], name='beta')
            # beta = concatenate([alpha,pred1, pred2], name='beta')
            # pred3 = Dense(number_classes[2], activation='softmax', name='L3_output')(beta)
            pred3 = Dense(number_classes[2], activation='softmax', name='L3_output')(alpha)
            model = Model(inputs, outputs=[pred1, pred2, pred3])
            model.compile(
                loss={'L1_output': LOSS, 'L2_output': LOSS, 'L3_output': LOSS},
                loss_weights={'L1_output': 0.1, 'L2_output': 0.1, 'L3_output': 1},
                optimizer=optimizers.Adam(learning_rate=LEARNING_RATE), metrics=['acc'])

        else:
            preds = Dense(number_classes[-1], activation='softmax')(alpha)
            model = Model(inputs, preds)
            model.compile(loss=LOSS, optimizer=optimizers.Adam(learning_rate=LEARNING_RATE), metrics=['acc'])

    if not muted: model.summary()
    return model


def split_test_data(block, gamma, predicted_label_L1_beam, features, num):
    features_split = [[[] for j in range(len(features))] for i in range(num + 1)]
    for idx_fea in range(len(features)):
        for idx_twt in range(features[0].shape[0]):
            if block[idx_twt] > gamma:
                features_split[-1][idx_fea].append(features[idx_fea][idx_twt])
            else:
                for label in predicted_label_L1_beam[idx_twt]:
                    features_split[label][idx_fea].append(features[idx_fea][idx_twt])
    features_split = [[np.asarray(i) for i in x] for x in features_split]
    return features_split


def getAccuracyK(predictions, labels, k):
    accurateCount = 0
    totalCount = predictions.shape[0]
    for i in range(totalCount):
        sorted = np.argsort(predictions[i])
        topK = sorted[-k:]
        if (labels[i] in topK):
            accurateCount = accurateCount + 1
    accuracy = accurateCount / totalCount
    return accuracy


def getDistance(pred_labels, true_labels, DATA):
    dfPOI = pd.read_csv('datasets/poi-mel-ex.csv')
    label_col = 'poiID'
    dists = [haversine(i, j, dfPOI, label_col) for i, j in zip(pred_labels, true_labels)]
    return np.average(dists), np.median(dists)


def haversine(i, j, dfPOI, label_col):
    lon1 = float(dfPOI.loc[(dfPOI[label_col] == i)]['long'])
    lat1 = float(dfPOI.loc[(dfPOI[label_col] == i)]['lat'])
    lon2 = float(dfPOI.loc[(dfPOI[label_col] == j)]['long'])
    lat2 = float(dfPOI.loc[(dfPOI[label_col] == j)]['lat'])
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371000
    return c * r


def evaluate_hierTagger(LOG_FILE, DATA, L1_model, L2_models, test_text, test_tc, test_time, test_labels, BATCH_SIZE,
                      number_classes, gamma, beam):
    print('Evaluating hierTagger:')
    pred_L1 = L1_model.predict(test_text + test_tc + test_time, batch_size=BATCH_SIZE, verbose=1)
    predicted_label_L1 = [np.argmax(pred_L1[i]) for i in range(pred_L1.shape[0])]
    block = [-1 / log(number_classes[0]) * np.sum([x * log(x) for x in pred_L1[i]]) for i in range(pred_L1.shape[0])]

    predicted_label_L1_beam = [np.argsort(x)[-beam:] for x in pred_L1]
    test_text_L2 = split_test_data(block, gamma, predicted_label_L1_beam, test_text, number_classes[0])
    test_tc_L2 = split_test_data(block, gamma, predicted_label_L1_beam, test_tc, number_classes[0])
    test_time_L2 = split_test_data(block, gamma, predicted_label_L1_beam, test_time, number_classes[0])
    test_labels_L2 = split_test_data(block, gamma, predicted_label_L1_beam, [test_labels[-1]], number_classes[0])
    test_labels_L2 = [x[0] for x in test_labels_L2]

    block_info = 'gamma: {}, test tweets: {}, Blocked tweets: {}'.format(gamma, test_text[0].shape[0],
                                                                         len(test_labels_L2[-1]))
    print(block_info)

    predictions = []
    for i in range(number_classes[0]):
        if i not in set(predicted_label_L1):
            predictions.append(None)
        elif test_text_L2[i][0].shape[0] == 0:
            predictions.append(None)
            print('Sub-classifier {} has no corresponding tweets'.format(i))
            continue
        else:
            predictions.append(L2_models[i].predict(test_text_L2[i] + test_tc_L2[i] + test_time_L2[i], batch_size=BATCH_SIZE, verbose=1))

    print('L2 grand classifier:')
    if test_text_L2[-1][0].shape[0] == 0:
        print('no corresponding tweets')
    else:
        pred_L2_grand = L2_models[-1].predict(test_text_L2[-1] + test_tc_L2[-1] + test_time_L2[-1], batch_size=BATCH_SIZE, verbose=1)
        predictions.append(pred_L2_grand)

    predictions_combine = []
    idx = [0 for i in range(number_classes[0] + 1)]
    for m, label in enumerate(predicted_label_L1_beam):
        if block[m] > gamma:
            predictions_combine.append(predictions[-1][idx[-1]])
            idx[-1] += 1
        else:
            temp = [predictions[x][idx[x]] for x in label]
            for i in label: idx[i] += 1
            result = []
            for k in range(np.max([len(x) for x in temp])):
                result.append(np.max([x[k] for x in temp if x.shape[0] > k]))
            predictions_combine.append(np.asarray(result))

    pred_L2 = [np.argmax(x) for x in predictions_combine]
    predictions_combine = np.asarray(predictions_combine)

    acc_L1 = [getAccuracyK(pred_L1, test_labels[0], 1), getAccuracyK(pred_L1, test_labels[0], 5),
              getAccuracyK(pred_L1, test_labels[0], 10), getAccuracyK(pred_L1, test_labels[0], 20)]
    macro_L1 = [f1_score(test_labels[0], predicted_label_L1, average='macro'),
                precision_score(test_labels[0], predicted_label_L1, average='macro', zero_division=1),
                recall_score(test_labels[0], predicted_label_L1, average='macro', zero_division=1)]

    acc_L2 = [getAccuracyK(predictions_combine, test_labels[-1], 1),
              getAccuracyK(predictions_combine, test_labels[-1], 5),
              getAccuracyK(predictions_combine, test_labels[-1], 10),
              getAccuracyK(predictions_combine, test_labels[-1], 20)]
    macro_L2 = [f1_score(test_labels[-1], pred_L2, average='macro'),
                precision_score(test_labels[-1], pred_L2, average='macro', zero_division=1),
                recall_score(test_labels[-1], pred_L2, average='macro', zero_division=1)]
    meanDist, medianDist = getDistance(pred_L2, test_labels[-1], DATA)
    performance_L1 = 'Level 0\nacc1, acc5, acc10, acc20, F1, precision, recall\n{:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}' \
        .format(acc_L1[0], acc_L1[1], acc_L1[2], acc_L1[3], macro_L1[0], macro_L1[1], macro_L1[2])
    performance_L3 = 'Level 2\nacc1, acc5, acc10, acc20, meanDistError\n{:.4}, {:.4}, {:.4}, {:.4}, {:.4}\nmedianDistError, F1, precision, recall\n{:.4}, {:.4}, {:.4}, {:.4}' \
        .format(acc_L2[0], acc_L2[1], acc_L2[2], acc_L2[3], meanDist, medianDist, macro_L2[0], macro_L2[1], macro_L2[2])
    print(performance_L1)
    print(performance_L3)
    with open(LOG_FILE, 'a') as f:
        f.write(block_info + '\n')
        f.write(performance_L1 + '\n')
        f.write(performance_L3 + '\n')
        f.write(
            '--------------------------------------------------------------------------------------------------------------------------------------------------------------\n')


def evaluate_Tagger_transTagger(LOG_FILE, DATA, model, test_text, test_tc, test_time, test_labels, BATCH_SIZE):
    print('Evaluating model:')
    preds = model.predict(test_text + test_tc + test_time, batch_size=BATCH_SIZE, verbose=1)
    labels = [np.argmax(preds[i]) for i in range(preds.shape[0])]
    meanDist, medianDist = getDistance(labels, test_labels[-1], DATA)
    results = [getAccuracyK(preds, test_labels[-1], 1), getAccuracyK(preds, test_labels[-1], 5),
               getAccuracyK(preds, test_labels[-1], 10), getAccuracyK(preds, test_labels[-1], 10),
               meanDist, medianDist,
               f1_score(test_labels[-1], labels, average='macro'),
               precision_score(test_labels[-1], labels, average='macro', zero_division=1),
               recall_score(test_labels[-1], labels, average='macro', zero_division=1),
               ]
    performance = 'acc1, acc5, acc10, acc20, meanDist\n{:.4}, {:.4}, {:.4}, {:.4}, {:.4}\nmedianDist, F1, precision, recall\n{:.4}, {:.4}, {:.4}, {:.4}'. \
        format(results[0], results[1], results[2], results[3], results[4], results[5], results[6], results[7],
               results[8])
    print(performance)
    with open(LOG_FILE, 'a') as f:
        f.write(performance + '\n')
        f.write(
            '--------------------------------------------------------------------------------------------------------------------------------------------------------------\n')


def evaluate_mtlTagger(LOG_FILE, DATA, model, test_text, test_tc, test_time, test_labels, BATCH_SIZE):
    print('Evaluating mtlTagger:')
    preds = model.predict(test_text + test_tc + test_time, batch_size=BATCH_SIZE, verbose=1)
    resultss = []
    count = 0
    for pre, gt in zip(preds, test_labels):
        label = [np.argmax(pre[i]) for i in range(pre.shape[0])]
        if count < 2:
            temp = [getAccuracyK(pre, gt, 1), getAccuracyK(pre, gt, 5),
                    getAccuracyK(pre, gt, 10), getAccuracyK(pre, gt, 20),
                    f1_score(gt, label, average='macro'),
                    precision_score(gt, label, average='macro', zero_division=1),
                    recall_score(gt, label, average='macro', zero_division=1)]
        else:
            meanDist, medianDist = getDistance(label, gt, DATA)
            temp = [getAccuracyK(pre, gt, 1), getAccuracyK(pre, gt, 5),
                    getAccuracyK(pre, gt, 10), getAccuracyK(pre, gt, 20),
                    meanDist, medianDist,
                    f1_score(gt, label, average='macro'),
                    precision_score(gt, label, average='macro', zero_division=1),
                    recall_score(gt, label, average='macro', zero_division=1)]
        count += 1
        resultss.append(temp)
    with open(LOG_FILE, 'a') as f:
        for i, results in enumerate(resultss):
            if i < 2:
                performance = 'Level {}\nacc1, acc5, acc10, acc20, F1, precision, recall\n{:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}'. \
                    format(i, results[0], results[1], results[2], results[3], results[4], results[5], results[6])
            else:
                performance = 'Level {}\nacc1, acc5, acc10, acc20, meanDist\n{:.4}, {:.4}, {:.4}, {:.4}, {:.4}\nmedianDist, F1, precision, recall\n{:.4}, {:.4}, {:.4}, {:.4}'. \
                    format(i, results[0], results[1], results[2], results[3], results[4], results[5], results[6],
                           results[7], results[8])
            print(performance)
            f.write(performance + '\n')
        f.write(
            '--------------------------------------------------------------------------------------------------------------------------------------------------------------\n')

