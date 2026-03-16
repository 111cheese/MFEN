import itertools
import tensorflow as tf
from filter_xiugai import GuidedFilter
import visualize as vis
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from config import MNF_data, rpca_data, two_improve_data, basic_data
import os
import matplotlib.pyplot as plt

def test_gray():
    toc1 = time.clock()
    input1 = MNF_data.data_path
    input_hsi1 = sio.loadmat(os.path.join(input1, 'data_mnfpavialyj.mat'))
    input_hsi1 = input_hsi1['M']
    input_hsi1=input_hsi1[:,:,0:45]
    array_x1 = input_hsi1.reshape(207400, 45)

    #单一变量对比实验，不使用mnf方法时将这使用这部分输入
    # input_hsi1 = sio.loadmat('D:/workspace/HSI_data/PaviaU.mat')
    # input_hsi1=input_hsi1['paviaU']
    # array_x1 = input_hsi1.reshape(207400, 103)

    pca = PCA(n_components=20)
    array_x2 = pca.fit_transform(array_x1)
    pca_hsi1 = array_x2.reshape(input_hsi1.shape[0], input_hsi1.shape[1], array_x2.shape[1])
    #image = pca_hsi1[:, :, 0]


    # input_hsi = sio.loadmat(os.path.join(basic_data.data_path,'PaviaU.mat'))
    # input_hsi = input_hsi['paviaU']
    # array_x1 = input_hsi.reshape(207400, 103)
    # ICA = FastICA(n_components=3)
    # array_x2 = ICA.fit_transform(array_x1)
    # ICA_hsi = array_x2.reshape(input_hsi.shape[0], input_hsi.shape[1], array_x2.shape[1])
    RPCA = sio.loadmat(os.path.join(rpca_data.data_path, 'data_rpcapaviaU1.mat'))
    RPCA = RPCA['M2']
    RPCA = RPCA[:, :, 0:103]
    rpca_x1 = RPCA.reshape(207400, 103)
    rpca_x2 = rpca_x1.reshape(RPCA.shape[0], RPCA.shape[1], rpca_x1.shape[1])
    print(rpca_x1.shape)
    print(rpca_x2.shape)

    image = rpca_x2[:, :, 0]
    radius = [2, 4,6]
    eps = [0.0001]
    combs = list(itertools.product(radius, eps))
    vis.plot_single(image, title='origin21')
    for i in range(0,20):
        for r, e in combs:
            GF = GuidedFilter(image, radius=r, eps=e)#原图，半径及eps
            x1 = GF.filter(pca_hsi1[:, :, i]).reshape(610, 340, 1)
            if r == 2:
               x=x1
            else:
               x=tf.concat([x,x1],2)
        if i==0:
           b=x
        else:
           b=tf.concat([b,x],2)
    sess = tf.Session()
    b = sess.run(b)
    tic1 = time.clock()
    print('Training Time: ', tic1 - toc1)
    return b

im=test_gray()
plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
plt.axis('off')
plt.figure(figsize=(im.shape[1] / 10, im.shape[0] / 10), dpi=100)  # set size
plt.xlim(0, 340)
plt.ylim(610, 0)
layer_slice = im[:, :, 0]
# gray_layer = 0.299 * layer_slice + 0.587 * layer_slice + 0.114 * layer_slice
plt.imshow(layer_slice, cmap='gray')
plt.show()
plt.close()
# sio.savemat(os.path.join(two_improve_data.data_path,'pavia_0110/upgf15_5.mat'), mdict={'upgf15_5': im})
#sio.savemat('E:/Dataset/Attention_ablash/'+'upgf15_5.mat', mdict={'upgf15_5': im})