import os
from pathlib import Path
from data_helper import data_processor, data_split, load_model_jh, save_model_jh, log_history
from model_builder import build_transTagger, evaluate_hierTagger, evaluate_Tagger_transTagger, evaluate_mtlTagger

MODEL_TYPE = 'mtlTagger'  # Tagger, transTagger
DATA = 'Twitter-Mel'  # Twitter-Mel, Flickr
INPUTS_REP = ['text', 'text']  # {text, 1hot} * {text, 1hot, uniform}
BERT_VER = 'Tiny'

BATCH_SIZE = 128  # 8, 16, 32, 64, 128
LEARNING_RATE = 3e-4  # 3e-4, 1e-4, 5e-5, 3e-5, default 1e-3
ENCODER = [3, 64, 1300, 0]
VALIDATION_SPLIT = 0.1
EPOCHS = 1
MAX_SEQUENCE_LENGTH = 100 if DATA == 'Flickr' else 0
BERT_TRAINABLE = True
ifPICKLE = True
POSITION = 'con'  # con, add
LOSS = 'sparse_categorical_crossentropy'

OUTPUT_PATH = './output'
MODEL_PATH = os.path.join(OUTPUT_PATH, 'models')

Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = f'{MODEL_TYPE}-{DATA}-{INPUTS_REP[0]}-{INPUTS_REP[1]}-{BERT_VER}'
LOG_FILE = f'{OUTPUT_PATH}/results.txt'
MODEL_NAME = f'{MODEL_PATH}/{OUTPUT_FILE}'

anno = f'{MODEL_TYPE}, {DATA}, inputs: {INPUTS_REP}, BERT version: {BERT_VER}, \nbatch size: {BATCH_SIZE}, learning rate: {LEARNING_RATE}, encoders: {ENCODER}, position encoding: {POSITION}'
print(anno)

[train_text, val_text, test_text, train_tc, val_tc, test_tc, train_time, val_time, test_time, train_labels, val_labels, test_labels, number_classes, counts_text, counts_unique_tc, counts_unique_time, correla_matircs] = \
    data_processor(DATA, INPUTS_REP, MODEL_TYPE, BERT_VER, MAX_SEQUENCE_LENGTH, VALIDATION_SPLIT, OUTPUT_PATH)

if MODEL_TYPE in ['Tagger', 'transTagger']:
    if os.path.exists(MODEL_NAME + '.json') & ifPICKLE: model = load_model_jh(MODEL_NAME, LOSS, LEARNING_RATE, MODEL_TYPE)
    else:
        print('Build model.')
        model = build_transTagger(INPUTS_REP, MODEL_TYPE, BERT_VER, BERT_TRAINABLE, POSITION, ENCODER, LEARNING_RATE, LOSS,
                                number_classes, counts_text, counts_unique_tc, counts_unique_time, correla_matircs, muted=True)
        history = model.fit(train_text + train_tc + train_time, train_labels[-1], batch_size=BATCH_SIZE,
                                epochs=EPOCHS, validation_data=(val_text + val_tc + val_time, val_labels[-1]))
        log_history(LOG_FILE, OUTPUT_FILE, anno, history, MODEL_TYPE)
        save_model_jh(model, MODEL_NAME)
    evaluate_Tagger_transTagger(LOG_FILE, DATA, model, test_text, test_tc, test_time, test_labels, BATCH_SIZE)

elif MODEL_TYPE == 'mtlTagger':
    if len(train_labels)==3: print('Hierarchical labels collected!')
    else: print('Error!') 
    if os.path.exists(MODEL_NAME + '.json') & ifPICKLE: model = load_model_jh(MODEL_NAME, LOSS, LEARNING_RATE, MODEL_TYPE)
    else:
        print('Build model.')
        model = build_transTagger(INPUTS_REP, MODEL_TYPE, BERT_VER, BERT_TRAINABLE, POSITION, ENCODER, LEARNING_RATE, LOSS,
                                number_classes, counts_text, counts_unique_tc, counts_unique_time, correla_matircs, muted=True)
        history = model.fit(train_text + train_tc + train_time,
                            {'L1_output': train_labels[0], 'L2_output': train_labels[1], 'L3_output': train_labels[2]},
                            batch_size=BATCH_SIZE, epochs=EPOCHS,
                            validation_data=(val_text + val_tc + val_time, {'L1_output': val_labels[0], 'L2_output': val_labels[1], 'L3_output': val_labels[2]}))
        log_history(LOG_FILE, OUTPUT_FILE, anno, history, MODEL_TYPE)
        save_model_jh(model, MODEL_NAME)
        evaluate_mtlTagger(LOG_FILE, DATA, model, test_text, test_tc, test_time, test_labels, BATCH_SIZE)

elif MODEL_TYPE == 'hierTagger':
    if len(train_labels)==3: print('Hierarchical labels collected!')
    else: print('Error!') 
    train_text_L2, train_tc_L2, train_time_L2, val_text_L2, val_time_L2, val_tc_L2, train_labels_L2, val_labels_L2, number_classes = data_split(
        train_text, train_tc, train_time, val_text, val_tc, val_time, train_labels, val_labels, number_classes)
    MODEL_NAME_L1, MODEL_NAME_L2, MODEL_NAME_grand = MODEL_NAME + '-L1', MODEL_NAME + '-L2', MODEL_NAME + '-grand'
    if (os.path.exists(MODEL_NAME_L1 + '.json')) & (ifPICKLE): L1_model = load_model_jh(MODEL_NAME_L1, LOSS, LEARNING_RATE, 'Level 1')
    else:
        print('Build model: Level 1.')
        L1_model = build_transTagger(INPUTS_REP, MODEL_TYPE, BERT_VER, BERT_TRAINABLE, POSITION, ENCODER, LEARNING_RATE,
                                   LOSS, [number_classes[0]], counts_text, counts_unique_tc, counts_unique_time, correla_matircs)
        L1_history = L1_model.fit(train_text + train_tc + train_time, train_labels[0], batch_size=BATCH_SIZE,
                                  epochs=EPOCHS, validation_data=(val_text + val_tc + val_time, val_labels[0]))
        log_history(LOG_FILE, OUTPUT_FILE, anno, L1_history, 'L1 history: ')
        save_model_jh(L1_model, MODEL_NAME_L1)
    L2_models = []
    for i in range(number_classes[0]):
        print('#classes of L2-{}: {}'.format(i, number_classes[-1][i]))
        if (number_classes[-1][i] <= 1):
            print('SKIP: ', i)
            L2_models.append(None)
        else:
            MODEL_NAME = '{}-{}'.format(MODEL_NAME_L2, i)
            if os.path.exists(MODEL_NAME + '.json') & ifPICKLE: model = load_model_jh(MODEL_NAME, LOSS, LEARNING_RATE, '')
            else:
                print('Build model Level 2: {}.'.format(i))
                model = build_transTagger(INPUTS_REP, MODEL_TYPE, BERT_VER, BERT_TRAINABLE, POSITION, ENCODER, LEARNING_RATE,
                                        LOSS, [number_classes[-1][i]], counts_text, counts_unique_tc, counts_unique_time, correla_matircs, muted=True)
                history = model.fit(train_text_L2[i] + train_tc_L2[i] + train_time_L2[i], train_labels_L2[i], batch_size=BATCH_SIZE,
                                    epochs=EPOCHS, validation_data=(val_text_L2[i] + val_tc_L2[i] + val_time_L2[i], val_labels_L2[i]))
                log_history(LOG_FILE, OUTPUT_FILE, anno, history, 'L2-{} history: '.format(i), muted=True)
                save_model_jh(model, MODEL_NAME)
            L2_models.append(model)
    print('#classes of L2-grand: {}'.format(number_classes[1]))
    if os.path.exists(MODEL_NAME_grand + '.json') & ifPICKLE: model = load_model_jh(MODEL_NAME_grand, LOSS, LEARNING_RATE, 'grand')
    else:
        print('Build L2 grand classifier.')
        model = build_transTagger(INPUTS_REP, MODEL_TYPE, BERT_VER, BERT_TRAINABLE, POSITION, ENCODER, LEARNING_RATE, LOSS,
                                [number_classes[-2]], counts_text, counts_unique_tc, counts_unique_time, correla_matircs, muted=True)
        history = model.fit(train_text + train_tc + train_time, train_labels[-1], batch_size=BATCH_SIZE, epochs=EPOCHS,
                            validation_data=(val_text + val_tc + val_time, val_labels[-1]))
        log_history(LOG_FILE, OUTPUT_FILE, anno, history, 'L2-grand history: ', muted=True)
        save_model_jh(model, MODEL_NAME_grand)
    L2_models.append(model)
    evaluate_hierTagger(LOG_FILE, DATA, L1_model, L2_models, test_text, test_tc, test_time, test_labels, BATCH_SIZE, number_classes, 0.01, 1)
