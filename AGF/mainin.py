import itertools
import numpy as np
import tensorflow as tf
from filter_xiugai import GuidedFilter
import visualize as vis
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
import time
import scipy.io as sio
import scipy.io as io
from config import MNF_data, rpca_data, two_improve_data, basic_data
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing

def test_gray():
    toc1 = time.clock()
    # 利用MNF得到输入图像
    input1 = MNF_data.data_path
    input_hsi1 = sio.loadmat(os.path.join(input1, 'data_mnfindianlyj.mat'))
    input_hsi1 = input_hsi1['M']
    input_hsi1=input_hsi1[:,:,0:50]
    array_x1 = input_hsi1.reshape(21025, 50)

    pca = PCA(n_components=10)
    array_x2 = pca.fit_transform(array_x1)
    pca_hsi1 = array_x2.reshape(input_hsi1.shape[0], input_hsi1.shape[1], array_x2.shape[1])
    print( pca_hsi1.shape)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
    plt.axis('off')
    plt.figure(figsize=(pca_hsi1.shape[1] / 10, pca_hsi1.shape[0] / 10), dpi=100)  # set size
    plt.xlim(0, 145)
    plt.ylim(145, 0)
    layer_slice = input_hsi1[:, :, 0]
    plt.imshow(layer_slice, cmap='gray')
    plt.show()
    plt.close()

    #*****引导图像用RPCA的
    RPCA = sio.loadmat(os.path.join(rpca_data.data_path, 'data_rpcaindian.mat'))
    RPCA = RPCA['M1']
    RPCA = RPCA[:, :, 0:200]
    rpca_x1 = RPCA.reshape(21025, 200)
    rpca_x2 = rpca_x1.reshape(RPCA.shape[0], RPCA.shape[1], rpca_x1.shape[1])
    print(rpca_x1.shape)
    print(rpca_x2.shape)

    #利用引导滤波和引导图像对输入图像进行引导滤波
    image = rpca_x2[:, :, 0]  #引导滤波

    radius = [2,4,6]
    eps = [0.0001]
    combs = list(itertools.product(radius, eps))
    vis.plot_single(image, title='origin21')
    for i in range(0, 10):
        for r, e in combs:
            GF = GuidedFilter(image, radius=r, eps=e)#引导图，半径及eps
            x1 = GF.filter(pca_hsi1 [:, :, i]).reshape(145, 145, 1) #输入图像
            if r == 2:
               x = x1
               #x=tf.concat([x1,y1],2)
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
# plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
# plt.axis('off')
# plt.figure(figsize=(im.shape[1] / 10, im.shape[0] / 10), dpi=100)  # set size
# plt.xlim(0, 145)
# plt.ylim(145, 0)
# layer_slice = im[:, :, 0]
# plt.imshow(layer_slice, cmap='gray')
# plt.show()
# plt.close()
sio.savemat(os.path.join(two_improve_data.data_path,'indian_0110/gfnoPCA_1.mat'), mdict={'gfnoPCA_1': im})