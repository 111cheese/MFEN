# -*- coding: utf-8 -*-
import sys
#sys.path.append("D:/workspace/lyj-chaofenbian")
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, Conv3D, MaxPooling3D, ZeroPadding3D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical
from sklearn.decomposition import PCA
from keras.optimizers import Adam, SGD, Adadelta, RMSprop, Nadam
import keras.callbacks as kcallbacks
from keras.regularizers import l2
import time
import collections
from sklearn import metrics, preprocessing
import zeroPadding,  doPCA, modelStatsRecord, averageAccuracy, Model_multi_new_UP, Model_multi_batch17
from keras.models import load_model
from config import basic_data, modelsave, two_improve_data, zishiying_data
import os
#
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def indexToAssignment(index_, Row, Col, pad_length):
    new_assign = {}
    new_assign_1 = []
    for counter, value in enumerate(index_):
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [(assign_0, assign_1)]
    for i in range(len(index_)):
        new_assign_1 += new_assign[i]
    #np.random.shuffle(new_assign)
    return new_assign_1


def loc(index_, Row, Col):
    new_assign = {}
    new_assign_1=[]
    for counter, value in enumerate(index_):
        assign_0 = value // Col
        assign_1 = value % Col
        new_assign[counter] = [(assign_0, assign_1)]
    for i in range(len(index_)):
        new_assign_1+=new_assign[i]
    #np.random.shuffle(new_assign)
    return new_assign_1

def assignmentToIndex(assign_0, assign_1, Row, Col):
    new_index = assign_0 * Col + assign_1
    return new_index


def selectNeighboringPatch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row - ex_len, pos_row + ex_len + 1), :]
    selected_patch = selected_rows[:, range(pos_col - ex_len, pos_col + ex_len + 1)]
    return selected_patch


def sampling(proptionVal, groundTruth):  # divide dataset into train and test datasets
    labels_loc = {}
    train = {}
    test = {}
    m = max(groundTruth)
    train_indices = []
    test_indices = []
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]#ravel为拉伸成一维数据
        # indices中为标签为1的样本的位置数
        # 每一类的样本数
        np.random.shuffle(indices)
        labels_loc[i] = indices
        nb_val = int(proptionVal * len(indices))
        train[i] = indices[:-nb_val]
        test[i] = indices[-nb_val:]
    #    whole_indices = []
    # 将所有的训练样本存到train集合中，将所有的测试样本存到test集合中

    for i in range(m):
        #        whole_indices += labels_loc[i]
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    print('测试样本个数：',len(test_indices))
    print('训练样本个数：',len(train_indices))
    return train_indices, test_indices


def MS_model():
    model = Model_multi_batch17.LSTMMCNN_RS(pca_dimension, nb_features, img_rows, img_cols)

    return model

# 加载数据
# 修正的Indian pines数据集
mat_data = sio.loadmat('/mnt/所有对比方法/WHU_Hi_LongKou.mat')
data_IN = mat_data['WHU_Hi_LongKou']
# 标签数据
mat_gt = sio.loadmat('/mnt/所有对比方法/WHU_Hi_LongKou_gt.mat')
gt_IN = mat_gt['WHU_Hi_LongKou_gt']

print('data_IN.shape:',data_IN.shape)
print('gt_IN.shape:',gt_IN.shape)
new_gt_IN = gt_IN
guided_data = sio.loadmat(os.path.join('./results/', 'LK_filtered_4.mat'))
guided = guided_data['filtered_4']

batch_size = 64
nb_classes = 9
nb_epoch = 350 # 400
img_rows, img_cols = 17, 17 # 27, 27
patience = 200 #200次不更新就停止

INPUT_DIMENSION = data_IN.shape[2]
pca_dimension = 30


TOTAL_SIZE = 204542
VAL_SIZE = 2051  # 1%
# VAL_SIZE = 1028  # 0.5%
TRAIN_SIZE = 2051
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
VALIDATION_SPLIT = 0.99  # 1% for trainnig and 80% for validation and testing


img_channels = guided.shape[2]
PATCH_LENGTH = 8  #如果是7*7的patch块儿，那么要padding上下左右3
PATCH_LENGTH1 = 8

gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]), )#21025

# 是否需要对输入进行PCA
pca = PCA(n_components=pca_dimension)
PcaInput = data_IN.reshape(data_IN.shape[0] * data_IN.shape[1], data_IN.shape[2])
data_pca = pca.fit_transform(PcaInput)
data_pca = preprocessing.scale(data_pca)
data_pca = data_pca.reshape(data_IN.shape[0], data_IN.shape[1], pca_dimension)
whole_data = data_pca
# whole_data = data_IN

padded_data1 = zeroPadding.zeroPadding_3D(guided, PATCH_LENGTH1)
padded_data = zeroPadding.zeroPadding_3D(whole_data, PATCH_LENGTH)

ITER = 1
CATEGORY = 9#类别

SLIC_train_data = np.zeros((TRAIN_SIZE, 2 * PATCH_LENGTH + 1, 2 * PATCH_LENGTH + 1, pca_dimension))#1031，7,7,200
SLIC_test_data = np.zeros((TEST_SIZE, 2 * PATCH_LENGTH + 1, 2 * PATCH_LENGTH + 1, pca_dimension))
frequency_train = np.zeros((TRAIN_SIZE, 2 * PATCH_LENGTH1 + 1, 2 * PATCH_LENGTH1 + 1,img_channels))
frequency_test = np.zeros((TEST_SIZE, 2 * PATCH_LENGTH1 + 1, 2 * PATCH_LENGTH1 + 1, img_channels))

num_PC = img_channels
nb_features = num_PC


KAPPA = []
OA = []
AA = []
TRAINING_TIME = []
TESTING_TIME = []
ELEMENT_ACC = np.zeros((ITER, CATEGORY))

# seeds = [1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229]

seeds = [1335]

for index_iter in range(ITER):
    print("# %d Iteration" % (index_iter + 1))

    # save the best validated model1
    # 使用easystopping通过一个动态阈值去选择最优的模型
    best_weights = './results/SSUN_1.hdf5'

    np.random.seed(seeds[index_iter])
    train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)

    # gt本身是标签类，从标签类中取出相应的标签 -1，转成one-hot形式
    y_train = gt[train_indices] - 1
    y_train_1 = y_train
    y_train = to_categorical(np.asarray(y_train))#相当于one-hot处理，asarray就是转换为矩阵

    y_test = gt[test_indices] - 1
    y_val_1 = y_test[-VAL_SIZE:]
    y_test = to_categorical(np.asarray(y_test))

    train_assign = indexToAssignment(train_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(train_assign)):
        SLIC_train_data[i] = selectNeighboringPatch(padded_data, train_assign[i][0], train_assign[i][1], PATCH_LENGTH)

    test_assign = indexToAssignment(test_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(test_assign)):
        SLIC_test_data[i] = selectNeighboringPatch(padded_data, test_assign[i][0], test_assign[i][1], PATCH_LENGTH)

    train_assignn = indexToAssignment(train_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH1)
    for i in range(len(train_assignn)):
        frequency_train[i] = selectNeighboringPatch(padded_data1, train_assignn[i][0], train_assignn[i][1], PATCH_LENGTH1)

    test_assignn = indexToAssignment(test_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH1)
    for i in range(len(test_assignn)):
        frequency_test[i] = selectNeighboringPatch(padded_data1, test_assignn[i][0], test_assignn[i][1], PATCH_LENGTH1)

    train_assign_0 = loc(train_indices, whole_data.shape[0], whole_data.shape[1])
    test_assign_0 = loc(test_indices, whole_data.shape[0], whole_data.shape[1])

    train_assign_1=train_assign_0[:TRAIN_SIZE]
    test_assign_1=test_assign_0[: -VAL_SIZE]
    val_assign=test_assign_0[-VAL_SIZE:]

    # 拿到了新的数据集进行reshpae之后，数据处理就结束了
    S_train = SLIC_train_data.reshape(SLIC_train_data.shape[0], SLIC_train_data.shape[1], SLIC_train_data.shape[2], pca_dimension)
    print(S_train.shape)#1031*7*7*200
    S_test_all = SLIC_test_data.reshape(SLIC_test_data.shape[0], SLIC_test_data.shape[1], SLIC_test_data.shape[2], pca_dimension)
    print(S_test_all.shape)#9218*7*7*200
    f_train = frequency_train.reshape(frequency_train.shape[0], frequency_train.shape[1], frequency_train.shape[2], img_channels)
    f_test_all = frequency_test.reshape(frequency_test.shape[0], frequency_test.shape[1], frequency_test.shape[2], img_channels)

    # 在测试数据集上进行验证和测试的划分
    S_val = S_test_all[-VAL_SIZE:]
    f_val = f_test_all[-VAL_SIZE:]
    #print(x_val.shape)
    y_val = y_test[-VAL_SIZE:]

    S_test = S_test_all[:-VAL_SIZE]
    f_test = f_test_all[:-VAL_SIZE]
    y_test = y_test[:-VAL_SIZE]
    print(y_test.shape)#8193*16----one-hot形式

    #加载模型
    ms_model = MS_model()

    earlyStopping6 = kcallbacks.EarlyStopping(monitor='val_joint_SOFTMAX_accuracy', patience=patience, verbose=1, mode='auto')
    saveBestModel6 = kcallbacks.ModelCheckpoint(best_weights, monitor='val_joint_SOFTMAX_accuracy', verbose=1,
                                                save_best_only=True,
                                                mode='auto')

    tic6 = time.perf_counter()

    # 1031*7*7*200 8193*7*7*200
    history= ms_model.fit(
        [S_train.reshape(S_train.shape[0], S_train.shape[1], S_train.shape[2], S_train.shape[3]), f_train.reshape(f_train.shape[0], f_train.shape[1], f_train.shape[2], f_train.shape[3])], [y_train,y_train,y_train],
        validation_data=([S_val.reshape(S_val.shape[0], S_val.shape[1], S_val.shape[2], S_val.shape[3]), f_val.reshape(f_val.shape[0], f_val.shape[1], f_val.shape[2], f_val.shape[3])], [y_val,y_val,y_val]),
        batch_size=batch_size,
        epochs=nb_epoch, shuffle=True, callbacks=[earlyStopping6, saveBestModel6])
    toc6 = time.perf_counter()

    print(history.history.keys())
    ms_model = load_model('./results/SSUN_1.hdf5')
    pred_test_list = []
    num_samples = TEST_SIZE-VAL_SIZE
    for i in range(0, num_samples, batch_size):
        # 获取当前批次的数据
        S_batch = S_test[i:min(i + batch_size, num_samples)]
        f_batch = f_test[i:min(i + batch_size, num_samples)]
        # 检查批次形状
        print(f"S_batch.shape: {S_batch.shape}, f_batch.shape: {f_batch.shape}")
        # 跳过空批次
        if S_batch.shape[0] == 0 or f_batch.shape[0] == 0:
            print("Skipping empty batch")
            continue
        # 重塑数据以匹配模型输入
        S_batch = S_batch.reshape(S_batch.shape[0], S_batch.shape[1], S_batch.shape[2], S_batch.shape[3])
        f_batch = f_batch.reshape(f_batch.shape[0], f_batch.shape[1], f_batch.shape[2], f_batch.shape[3])
        # 检查输入形状
        print(f"Model input shapes: {[input.shape for input in ms_model.inputs]}")
        # 进行预测
        pred_batch = ms_model.predict([S_batch, f_batch], verbose=0)
        print(f"pred_batch shape: {np.array(pred_batch).shape}")
        # 获取预测结果的类别索引
        pred_batch = np.argmax(pred_batch[0], axis=-1)
        # 将当前批次的预测结果存储到列表中
        pred_test_list.append(pred_batch)
    # 将所有批次的预测结果合并为一个数组
    pred_test = np.concatenate(pred_test_list)
    # 检查是否遗漏了样本
    assert len(pred_test) == num_samples, "Predicted samples do not match total samples!"
    # pred_test = ms_model.predict(
    #     [S_test.reshape(S_test.shape[0], S_test.shape[1], S_test.shape[2], S_test.shape[3]), f_test.reshape(f_test.shape[0], f_test.shape[1], f_test.shape[2], f_test.shape[3])],verbose=0)[0].argmax(axis=-1)
    collections.Counter(pred_test)
    gt_test = gt[test_indices] - 1
    #print(gt_test.shape)
    # 这是测试集，验证和测试还没有分开
    overall_acc = metrics.accuracy_score(gt_test[:-VAL_SIZE],pred_test)
    confusion_matrix = metrics.confusion_matrix(gt_test[:-VAL_SIZE],pred_test)
    each_acc, average_acc = averageAccuracy.AA_andEachClassAccuracy(confusion_matrix)
    kappa = metrics.cohen_kappa_score(gt_test[:-VAL_SIZE],pred_test)
    KAPPA.append(kappa)
    OA.append(overall_acc)
    AA.append(average_acc)
    TRAINING_TIME.append(toc6 - tic6)
    #TESTING_TIME.append(toc7 - tic7)
    ELEMENT_ACC[index_iter, :] = each_acc

    ac_list = []
    for i in range(len(confusion_matrix)):
        ac = confusion_matrix[i, i] / sum(confusion_matrix[i, :])
        ac_list.append(ac)
        print(i + 1, 'class:', '(', confusion_matrix[i, i], '/', sum(confusion_matrix[i, :]), ')', ac)
    print(' OA:', OA)
    print(' AA:', AA)
    print(' KAPPA:', KAPPA)
    # print('confusion_matrix:',confusion_matrix)
    print("training finished.")

    gt_test_1 = gt_test[: -VAL_SIZE]

    plot_max = np.zeros((550, 400))


    idx = 0
    for (i, j) in test_assign_1:
        # plot_max[i, j] = gt_test_1[idx] + 1
        plot_max[i, j] = pred_test[idx] + 1
        idx += 1

    idx = 0
    for (i, j) in train_assign_1:
        plot_max[i, j] = y_train_1[idx] + 1
        idx += 1
    idx = 0
    for (i, j) in val_assign:
        plot_max[i, j] = y_val_1[idx] + 1
        idx += 1

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
    plt.axis('off')
    # plt.subplots(figsize=(340,610))
    # matplotlib.rcParams['figure.figsize'] = [1.45, 1.45]
    plt.figure(figsize=(plot_max.shape[1] / 10, plot_max.shape[0] / 10), dpi=100)  # set size
    plt.xlim(0, 400)
    plt.ylim(550, 0)
    plt.pcolor(plot_max, cmap='jet')
    # plt.savefig(os.path.join('result', 'decode_map_' + NO_data + '.png'), format='png')
    save_path = './result/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_filename = os.path.join(save_path, 'SS_' + 'noGate' + '.png')
    plt.savefig(save_filename, format='png')
    plt.show()
    plt.close()
    print('decode map get finished')

modelStatsRecord.outputStats(KAPPA, OA, AA, ELEMENT_ACC,
                             TRAINING_TIME, TESTING_TIME,
                             './results/LK_new1%.txt')