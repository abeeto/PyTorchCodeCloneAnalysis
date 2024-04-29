import os, pickle
import pandas as pd
import numpy as np
import lightgbm
import librosa
import ntpath
import h5py
from sklearn.model_selection import train_test_split


def rectify_data(base_dir, meta_path, n_classes, size, experiments_path):
    meta_data = pd.read_csv(meta_path, delimiter=';')
    meta_data = meta_data[meta_data.cur_label!='disgust'][meta_data.cur_label!='excitement'][meta_data.cur_label!='fear'][meta_data.cur_label!='frustration'] \
            [meta_data.cur_label != 'oth'][meta_data.cur_label != 'surprise'][meta_data.cur_label != 'xxx']#[meta_data.cur_label != 'neutrality']

    rect_x, rect_y, ground_truth_sr = [], [], 8000

    # проходимся по каждому файлу и нарезаем его на куски по 16000 отчетов
    for idx, row in meta_data.loc[:,['cur_name','cur_label']].iterrows():
        x, sr = librosa.core.load(os.path.join(base_dir, 'data', row[0]), sr=None, mono=False, res_type='kaiser_best', dtype=np.float32)

        # проверка на соответствие sample_rate правильному значению
        # assert sr == ground_truth_sr, 'What the hell is going on with sample rate of your files?'

        x = list(x)
        y = row[1]

        # добавляем куски по 16000 с нахлестом 8000
        k1=0
        for k in range(0, len(x)-size, size//2):
            rect_x.append(tuple(x[k:k+size]))
            rect_y.append(y)
            k1=k
        for k2 in range(k1+size//2, len(x), size//2):
            element = x[k2:]
            element = np.pad(element, (0, size - len(element)), mode='constant')
            rect_x.append(tuple(element))
            rect_y.append(y)

    # вспомогательная функция, чтобы привести лист листов к матричному виду
    def make_array(x):
        new_x = [np.array(c) for c in x]

        array_x = np.vstack(new_x)
        return array_x

    final_x = make_array(rect_x)

    # переводим категориальные призаки к вещественным, чтобы было возможно производить расчеты,
    # а для обратного обращения создаем словарь
    dict_emo, reverse_dict_emo = {}, {}

    for idx,i in enumerate(np.unique(rect_y)):
        dict_emo[i] = idx
        reverse_dict_emo[idx] = i

    rect_y = [dict_emo[i] for i in rect_y]
    rect_y = np.array(rect_y).reshape(-1,1)

    rect_data = np.hstack((final_x, rect_y))

    # сохраняем эти матрицы
    if ntpath.basename(str(meta_path))=='meta_train.csv':
        h5f = h5py.File(os.path.join(experiments_path, 'x_train_{}cls.h5'.format(n_classes)), 'w')
        h5f.create_dataset('x_train', data=rect_data)
        h5f.close()
        print('train part is created')
    else:
        h5f = h5py.File(os.path.join(experiments_path, 'x_test_{}cls.h5'.format(n_classes)), 'w')
        h5f.create_dataset('x_test', data=rect_data)
        h5f.close()
        print('test part is created')



def get_raw_data(base_path, experiments_path, n_classes, size=16000):
    # загрузка посчитанных данных
    train_meta_path = os.path.join(base_path, 'meta_train.csv')
    test_meta_path = os.path.join(base_path, 'meta_test.csv')

    # подсчитать test и train выборки
    if os.path.isfile(os.path.join(experiments_path, 'x_test_{}cls.h5'.format(n_classes)))==False:
        test = rectify_data(base_path, test_meta_path, n_classes, size, experiments_path=experiments_path)
        train = rectify_data(base_path, train_meta_path, n_classes, size, experiments_path=experiments_path)

    # загрузить уже подсчитанные test и train выборки
    hf_train = h5py.File(os.path.join(experiments_path, 'x_train_{}cls.h5'.format(n_classes)), 'r')
    train = hf_train.get('x_train').value
    hf_train.close()

    hf_test = h5py.File(os.path.join(experiments_path, 'x_test_{}cls.h5'.format(n_classes)), 'r')
    test = hf_test.get('x_test').value
    hf_test.close()

    (N,W) = train.shape

    # x_train = train[:int(0.8*N), :16000]
    # y_train = train[:int(0.8*N), -1]
    # x_val = train[int(0.8*N):, :16000]
    # y_val = train[int(0.8*N):, -1]

    x_test = test[:, :size]
    y_test = test[:, -1]
    # x_train = train[:, :16000]
    # y_train = train[:, -1]
    x_train, x_val, y_train, y_val = train_test_split(train[:,:size], train[:,-1], test_size = 0.2, random_state = 42)
    # print('np.unique(y_train) ', np.unique(y_train, return_counts=True))
    # print('np.unique(y_val) ', np.unique(y_val, return_counts=True))
    # print('np.unique(y_test) ', np.unique(y_test, return_counts=True))

    return [x_train, x_val, x_test, y_train, y_val, y_test]



# def get_iemocap_raw2():
#     # загрузка посчитанных данных
#     base_path = r'C:\Users\kotov-d\Documents\BASES\IEMOCAP\iemocap'
#     train_meta_path = os.path.join(base_path, 'meta_train.csv')
#     test_meta_path = os.path.join(base_path, 'meta_test.csv')
#
#     # подсчитать test и train выборки
#     test = rectify_data2(base_path, test_meta_path)
#     train = rectify_data2(base_path, train_meta_path)
#
#
#     (N, W) = train.shape
#
#     np.random.shuffle(train)
#
#     x_train = train[:int(0.8 * N), :32000]
#     y_train = train[:int(0.8 * N), -1]
#     x_val = train[int(0.8 * N):, :32000]
#     y_val = train[int(0.8 * N):, -1]
#     x_test = test[:, :32000]
#     y_test = test[:, -1]
#     print('np.unique(y_train) ', np.unique(y_train, return_counts=True))
#     print('np.unique(y_test) ', np.unique(y_test, return_counts=True))
#
#     return [x_train, x_val, x_test, y_train, y_val, y_test]


# def get_iemocap_opensmile():
#     features_path = r'C:\Users\kotov-d\Documents\BASES\IEMOCAP\iemocap\feature\opensmile'
#
#
#     with open(os.path.join(features_path, 'x_train.pkl'), 'rb') as f:
#         x_train = pd.DataFrame(np.vstack(pickle.load(f)))
#     with open(os.path.join(features_path, 'x_test.pkl'), 'rb') as f:
#         x_test = pd.DataFrame(np.vstack(pickle.load(f)))
#     with open(os.path.join(features_path, 'y_train.pkl'), 'rb') as f:
#         y_train = pickle.load(f).loc[:, 'cur_label']
#     with open(os.path.join(features_path, 'y_test.pkl'), 'rb') as f:
#         y_test = pickle.load(f).loc[:, 'cur_label']
#
#
#     train = pd.concat([x_train, y_train], axis=1)
#     train = train[train.cur_label!='dis'][train.cur_label!='exc'][train.cur_label!='fea'][train.cur_label!='fru'] \
#                                             [train.cur_label != 'hap'][train.cur_label != 'oth'][train.cur_label != 'sur'][train.cur_label != 'xxx']
#     test = pd.concat([x_test, y_test], axis=1)
#     test = test[test.cur_label != 'dis'][test.cur_label != 'exc'][test.cur_label != 'fea'][test.cur_label != 'fru'] \
#         [test.cur_label != 'oth'][test.cur_label != 'sur'][test.cur_label != 'xxx']
#
#     train = train.sample(frac=1)
#
#     with open(r'C:\Users\kotov-d\Documents\TASKS\task#7\dictionaries.pkl', 'rb') as f:
#         [dict_emo, reverse_dict_emo] = pickle.load(f)
#
#
#     x_train = train.values[:int(0.8 * N), :-1].astype(np.float32)
#     y_train = train.values[:int(0.8 * N), -1]
#     y_train = np.array([dict_emo[i] for i in y_train]).reshape(-1,)
#
#     x_val = train.values[int(0.8 * N):, :-1].astype(np.float32)
#     y_val = train.values[int(0.8 * N):, -1]
#     y_val = np.array([dict_emo[i] for i in y_val]).reshape(-1,)
#
#     x_test = test.values[:, :-1].astype(np.float32)
#     y_test = test.values[:, -1]
#     y_test = np.array([dict_emo[i] for i in y_test]).reshape(-1,)
#
#
#     return [x_train, x_val, x_test, y_train, y_val, y_test]
#
#
#
#
# def rectify_data2(base_dir, meta_path):
#     meta_data = pd.read_csv(meta_path, delimiter=';')
#     meta_data = meta_data[meta_data.cur_label!='dis'][meta_data.cur_label!='exc'][meta_data.cur_label!='fea'][meta_data.cur_label!='fru'] \
#                                             [meta_data.cur_label != 'hap'][meta_data.cur_label != 'oth'][meta_data.cur_label != 'sur'][meta_data.cur_label != 'xxx']
#
#     rect_x, rect_y, ground_truth_sr = [], [], 16000
#
#     # проходимся по каждому файлу и нарезаем его на куски по 16000 отчетов
#     for idx ,row in meta_data.loc[:,['cur_name','cur_label']].iterrows():
#         x, sr = librosa.core.load(os.path.join(base_dir, 'data', row[0]), sr=None, mono=False, res_type='kaiser_best', dtype=np.float32)
#
#         # проверка на соответствие sample_rate правильному значению
#         if sr != ground_truth_sr:
#             print('What the hell is going on with sample rate of your files?')
#             raise Exception()
#
#         # чтобы быстрее считалось снизим память выделяемую под каждое число
#         x.astype(np.float16)
#
#         x = [k for k in x]
#         y = row[1]
#
#         # добавляем куски по 16000 с нахлестом 8000                         что тут за дерьмо происходит, Денис?
#         k1=0
#         for k in range(0, len(x)-32000, 8000):
#             rect_x.append(tuple(x[k:k+32000]))
#             rect_y.append(y)
#             k1=k
#         for k2 in range(k1+8000, len(x), 8000):
#             element = x[k2:]
#             element = np.pad(element, (0, 32000 - len(element)), mode='constant')
#             rect_x.append(tuple(element))
#             rect_y.append(y)
#
#         if idx%50==0 and idx!=0:
#             print(idx)
#
#     # вспомогательная функция, чтобы привести лист листов к матричному виду
#     def make_array(x):
#         new_x = [np.array(c) for c in x]
#
#         array_x = np.vstack(new_x)
#         return array_x
#
#     final_x = make_array(rect_x)
#
#
#     # переводим категориальные призаки к вещественным, чтобы было возможно производить расчеты,
#     # а для обратного обращения создаем словарь
#     dict_emo, reverse_dict_emo, mark = {}, {}, True
#     if mark==True:
#         for idx,i in enumerate(np.unique(rect_y)):
#             dict_emo[i] = idx
#             reverse_dict_emo[idx] = i
#             mark=False
#         with open(r'C:\Users\kotov-d\Documents\TASKS\task#7\dictionaries.pkl', 'wb') as f:
#             pickle.dump([dict_emo, reverse_dict_emo], f, protocol=2)
#     else:
#         with open(r'C:\Users\kotov-d\Documents\TASKS\task#7\dictionaries.pkl', 'rb') as f:
#             [dict_emo, reverse_dict_emo] = pickle.load(f)
#
#     rect_y = [dict_emo[i] for i in rect_y]
#     rect_y = np.array(rect_y).reshape(-1,1)
#
#     rect_data = np.hstack((final_x, rect_y))
#
#
#     return rect_data
#
#
# def get_data():
#     base_dir = r'C:\Users\kotov-d\Documents\bases'
#
#     omg_features_path = os.path.join(base_dir, 'omg', 'feature', 'opensmile')
#     iemocap_features_path = os.path.join(base_dir, 'iemocap', 'feature', 'opensmile')
#     mosei_features_path = os.path.join(base_dir, 'cmu_mosei', 'feature', 'opensmile')
#
#     def make_x_and_y_iemocap(x, y):
#         temp = pd.concat([pd.DataFrame(np.array(x)), y], axis=1)
#         temp = temp[temp['cur_label'] != 'xxx'][temp['cur_label'] != 'oth'][temp['cur_label'] != 'dis'][
#             temp['cur_label'] != 'fru'][temp['cur_label'] != 'exc'] \
#             [temp['cur_label'] != 'sur'][temp['cur_label'] != 'fea'][temp['cur_label'] != 'neu']
#         new_x, new_y = temp.iloc[:, :-1], temp.iloc[:, -1]
#         return [new_x, new_y]
#
#     with open(os.path.join(iemocap_features_path, 'x_train.pkl'), 'rb') as f:
#         iemocap_x_train = pickle.load(f)
#     with open(os.path.join(iemocap_features_path, 'x_test.pkl'), 'rb') as f:
#         iemocap_x_test = pickle.load(f)
#     with open(os.path.join(iemocap_features_path, 'y_train.pkl'), 'rb') as f:
#         iemocap_y_train = pickle.load(f).loc[:, 'cur_label']
#     with open(os.path.join(iemocap_features_path, 'y_test.pkl'), 'rb') as f:
#         iemocap_y_test = pickle.load(f).loc[:, 'cur_label']
#
#     [iemocap_x_train, iemocap_y_train] = make_x_and_y_iemocap(iemocap_x_train, iemocap_y_train)
#     [iemocap_x_test, iemocap_y_test] = make_x_and_y_iemocap(iemocap_x_test, iemocap_y_test)
#
#     def make_x_and_y_omg(x, y):
#         dict_emo = {'anger': 'ang', 'happy': 'hap', 'neutral': 'neu', 'surprise': 'sur', 'disgust': 'dis', 'sad': 'sad',
#                     'fear': 'fea'}
#         y = y.map(lambda x: dict_emo[x])
#         temp = pd.concat([pd.DataFrame(np.array(x)), y], axis=1)
#         temp = temp[temp['cur_label'] != 'dis'][temp['cur_label'] != 'sur'] \
#             [temp['cur_label'] != 'fea'][temp['cur_label'] != 'neu'].reset_index(drop=True)
#         new_x, new_y = temp.iloc[:, :-1], temp.iloc[:, -1]
#         return [new_x, new_y]
#
#     with open(os.path.join(omg_features_path, 'x_train.pkl'), 'rb') as f:
#         omg_x_train = pickle.load(f)
#     with open(os.path.join(omg_features_path, 'x_test.pkl'), 'rb') as f:
#         omg_x_test = pickle.load(f)
#     with open(os.path.join(omg_features_path, 'y_train.pkl'), 'rb') as f:
#         omg_y_train = pickle.load(f).loc[:, 'cur_label']
#     with open(os.path.join(omg_features_path, 'y_test.pkl'), 'rb') as f:
#         omg_y_test = pickle.load(f).loc[:, 'cur_label']
#
#     [omg_x_train, omg_y_train] = make_x_and_y_omg(omg_x_train, omg_y_train)
#     [omg_x_test, omg_y_test] = make_x_and_y_omg(omg_x_test, omg_y_test)
#
#     def make_x_and_y_mosei(x, y):
#         dict_emo = {'anger': 'ang', 'happiness': 'hap', 'surprise': 'sur', 'disgust': 'dis', 'sadness': 'sad',
#                     'fear': 'fea'}
#         y = y.map(lambda x: dict_emo[x])
#         temp = pd.concat([pd.DataFrame(np.array(x)), y], axis=1)
#         temp = temp[temp['cur_label'] != 'dis'][temp['cur_label'] != 'sur'] \
#             [temp['cur_label'] != 'fea'].reset_index(drop=True)
#         new_x, new_y = temp.iloc[:, :-1], temp.iloc[:, -1]
#         return [new_x, new_y]
#
#     with open(os.path.join(mosei_features_path, 'x_train.pkl'), 'rb') as f:
#         mosei_x_train = pickle.load(f)
#     with open(os.path.join(mosei_features_path, 'x_test.pkl'), 'rb') as f:
#         mosei_x_test = pickle.load(f)
#     with open(os.path.join(mosei_features_path, 'y_train.pkl'), 'rb') as f:
#         mosei_y_train = pickle.load(f).loc[:, 'cur_label']
#     with open(os.path.join(mosei_features_path, 'y_test.pkl'), 'rb') as f:
#         mosei_y_test = pickle.load(f).loc[:, 'cur_label']
#
#     [mosei_x_train, mosei_y_train] = make_x_and_y_mosei(mosei_x_train, mosei_y_train)
#     [mosei_x_test, mosei_y_test] = make_x_and_y_mosei(mosei_x_test, mosei_y_test)
#
#     # ==========================================================================
#     # take only top 100 features
#     # clf = LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.001, n_estimators=1000,
#     #                      objective=None, min_split_gain=0, min_child_weight=3, min_child_samples=10, subsample=0.8,
#     #                      subsample_freq=1, colsample_bytree=0.7, reg_alpha=0.3, reg_lambda=0, seed=17)
#     #
#     # clf.fit(iemocap_x_train, iemocap_y_train)
#     #
#     # with open(os.path.join(r'C:\Users\preductor\PycharmProjects\STC', 'clf' + '.pkl'), 'wb') as f:
#     #     clf = pickle.dump(clf, f, protocol=3)
#
#     with open(os.path.join(r'C:\Users\kotov-d\Documents\task#5', 'clf' + '.pickle'), 'rb') as f:
#         clf = pickle.load(f)
#
#     dict_importance = {}
#     for feature, importance in zip(range(len(clf.feature_importances_)), clf.feature_importances_):
#         dict_importance[feature] = importance
#
#     best_features = []
#
#     for idx, w in enumerate(sorted(dict_importance, key=dict_importance.get, reverse=True)):
#         if idx == 100:
#             break
#         best_features.append(w)
#
#     iemocap_x_train = iemocap_x_train.loc[:, best_features]
#     iemocap_x_test = iemocap_x_test.loc[:, best_features]
#     omg_x_train = omg_x_train.loc[:, best_features]
#     omg_x_test = omg_x_test.loc[:, best_features]
#     mosei_x_train = mosei_x_train.loc[:, best_features]
#     mosei_x_test = mosei_x_test.loc[:, best_features]
#
#     return [[iemocap_x_train, iemocap_x_test, omg_x_train, omg_x_test, mosei_x_train, mosei_x_test],
#         [iemocap_y_train, iemocap_y_test, omg_y_train, omg_y_test, mosei_y_train, mosei_y_test]]
