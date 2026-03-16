import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from skimage.segmentation import slic,mark_boundaries
from sklearn import preprocessing
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans


def SegmentsLabelProcess(labels):
    '''
    对labels做后处理，防止出现label不连续现象
    '''
    labels = np.array(labels, np.int64)
    H, W = labels.shape
    ls = list(set(np.reshape(labels, [-1]).tolist()))
    
    dic = {}
    for i in range(len(ls)):
        dic[ls[i]] = i
    
    new_labels = labels
    for i in range(H):
        for j in range(W):
            new_labels[i, j] = dic[new_labels[i, j]]
    return new_labels


class SLIC(object):
    def __init__(self, HSI,labels, n_segments=1000, compactness=20, max_iter=20, sigma=0, min_size_factor=0.3,
                 max_size_factor=2):
        self.n_segments = n_segments
        self.compactness = compactness
        self.max_iter = max_iter
        self.min_size_factor = min_size_factor
        self.max_size_factor = max_size_factor
        self.sigma = sigma
        # 数据standardization标准化,即提前全局BN
        height, width, bands = HSI.shape  # 原始高光谱数据的三个维度
        data = np.reshape(HSI, [height * width, bands])
        minMax = preprocessing.StandardScaler()
        data = minMax.fit_transform(data)
        self.data = np.reshape(data, [height, width, bands])
        self.labels=labels

    def hierarchical_clustering_refinement(self, segments, n_clusters=5):
        """
        基于分层聚类的超像素细化。
        参数:
        img -- 输入图像
        segments -- 初始超像素标签
        n_clusters -- 每个超像素的聚类数
        返回:
        refined_segments -- 分层聚类后的超像素标签
        """
        img = self.data
        refined_segments = np.zeros_like(segments)
        new_label = 0
        for label in np.unique(segments):
            mask = (segments == label)
            sub_img = img[mask]
            # 对子区域像素进行 KMeans 聚类
            if len(sub_img) >= n_clusters*2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                sub_segments = kmeans.fit_predict(sub_img)
            else:
                sub_segments = np.zeros(len(sub_img), dtype=int)
            # 将聚类结果映射回原图
            refined_segments[mask] = sub_segments + new_label
            new_label += sub_segments.max() + 1
        return refined_segments
    
    def get_Q_and_S_and_Segments(self):
        # 执行 SLCI 并得到Q(nxm),S(m*b)
        img = self.data
        (h, w, d) = img.shape
        # 计算超像素S以及相关系数矩阵Q
        segments = slic(img, n_segments=self.n_segments, compactness=self.compactness, max_num_iter=self.max_iter,
                        convert2lab=False,sigma=self.sigma, enforce_connectivity=True,
                        min_size_factor=self.min_size_factor, max_size_factor=self.max_size_factor,slic_zero=False)
        # segments = self.hierarchical_clustering_refinement(segments)

        # segments = felzenszwalb(img, scale=1,sigma=0.5,min_size=25)
        
        # segments = quickshift(img,ratio=1,kernel_size=5,max_dist=4,sigma=0.8, convert2lab=False)
        
        # segments=LSC_superpixel(img,self.n_segments)
        
        # segments=SEEDS_superpixel(img,self.n_segments)
        
        # 判断超像素label是否连续,否则予以校正
        if segments.max()+1!=len(list(set(np.reshape(segments,[-1]).tolist()))): segments = SegmentsLabelProcess(segments)
        self.segments = segments
        superpixel_count = segments.max() + 1
        self.superpixel_count = superpixel_count
        print("superpixel_count", superpixel_count)
        
        # ######################################显示超像素图片
        out = mark_boundaries(img[:,:,[0,1,2]], segments)
        # out = (img[:, :, [0, 1, 2]]-np.min(img[:, :, [0, 1, 2]]))/(np.max(img[:, :, [0, 1, 2]])-np.min(img[:, :, [0, 1, 2]]))
        plt.figure()
        plt.imshow(out)
        plt.show()
        
        segments = np.reshape(segments, [-1])
        S = np.zeros([superpixel_count, d], dtype=np.float32)
        Q = np.zeros([w * h, superpixel_count], dtype=np.float32)
        x = np.reshape(img, [-1, d])
        
        for i in range(superpixel_count):
            idx = np.where(segments == i)[0]
            count = len(idx)
            pixels = x[idx]
            superpixel = np.sum(pixels, 0) / count
            S[i] = superpixel
            Q[idx, i] = 1
        
        self.S = S
        self.Q = Q
        
        return Q, S , self.segments


    def get_A_my(self, neighbors_num=2, use_ksam=False, delta_ksam=0.55,
                              mul_spat=True, emphasize_smooth=True, sigma=0.174, beta=0.6):
        '''
        根据高光谱图像生成邻接矩阵
        :param num_node: 超像素节点总数_____self.superpixel_count
        :param hsi: 高光谱图像的特征矩阵 (num_node, feature_dim)____self.S
        :param neighbors_num: 每个节点的邻居数
        :param use_ksam: 是否使用 Kernel Spectral Angle Mapper (KSAM)
        :param delta_ksam: KSAM 的核宽参数
        :param mul_spat: 是否使用空间约束乘法
        :param emphasize_smooth: 是否对空间距离使用平滑约束
        :param sigma: RBF 核参数
        :param beta: 空间距离约束的权重因子
        :return: 超像素邻接矩阵 (num_node, num_node)
        '''
        # 1. 特征归一化
        hsi = self.S
        num_node = self.superpixel_count
        hsi_data = (hsi - np.min(hsi, axis=0)) / (np.max(hsi, axis=0) - np.min(hsi, axis=0) + 1e-8)
        # 2. 计算 Jensen-Shannon Divergence
        distribution = hsi_data / np.sum(hsi_data, axis=1, keepdims=True)  # Softmax 分布
        average_distribution = 0.5 * (distribution[:, None, :] + distribution[None, :, :])
        kl_p_m = np.sum(distribution[:, None, :] * (np.log(distribution[:, None, :] + 1e-8) -
                                                    np.log(average_distribution + 1e-8)), axis=2)
        kl_q_m = np.sum(distribution[None, :, :] * (np.log(distribution[None, :, :] + 1e-8) -
                                                    np.log(average_distribution + 1e-8)), axis=2)
        js_divergence = 0.5 * (kl_p_m + kl_q_m)
        distanceH = js_divergence + js_divergence.T  # 对称化
        # 3. 计算 KSAM 或余弦相似性
        if use_ksam:
            gamma = 1.0 / (2 * delta_ksam ** 2)
            phi = rbf_kernel(hsi_data, gamma=gamma)
        else:
            normed_hsi = hsi_data / np.linalg.norm(hsi_data, axis=1, keepdims=True)
            phi = np.clip(np.dot(normed_hsi, normed_hsi.T), -1 + 1e-8, 1 - 1e-8)
        spectral_similarity = np.sin(np.arccos(phi))
        distanceH *= spectral_similarity  # 与光谱相似性相乘
        # 4. 计算空间约束（Chebyshev 距离）
        indices = np.arange(num_node)[:, None]
        chebyshev_distance = np.max(np.abs(indices - indices.T), axis=-1)
        spatial_distance = np.log1p(chebyshev_distance) if emphasize_smooth else np.log(2 + chebyshev_distance)
        if mul_spat:
            distanceH *= spatial_distance
        else:
            distanceH += beta * spatial_distance
        # 5. 生成邻接矩阵
        A = np.zeros((num_node, num_node), dtype=np.float32)
        for i in range(num_node):
            # 找到每个节点的 k 个最近邻
            neighbors = np.argsort(distanceH[i])[:neighbors_num]
            for j in neighbors:
                A[i, j] = A[j, i] = np.exp(-distanceH[i, j] / sigma ** 2)

        return A

class AGF_SLIC(object):
    def __init__(self,data,labels,n_component):
        self.data=data
        self.init_labels=labels
        self.curr_data=data
        self.n_component=n_component
        self.height,self.width,self.bands=data.shape
        self.x_flatt=np.reshape(data,[self.width*self.height,self.bands])
        self.y_flatt=np.reshape(labels,[self.height*self.width])
        self.labes=labels
        
    def LDA_Process(self,curr_labels):
        '''
        :param curr_labels: height * width
        :return:
        '''
        curr_labels=np.reshape(curr_labels,[-1])
        idx=np.where(curr_labels!=0)[0]
        x=self.x_flatt[idx]
        y=curr_labels[idx]
        lda = LinearDiscriminantAnalysis()#n_components=self.n_component
        lda.fit(x,y-1)
        X_new = lda.transform(self.x_flatt)
        return np.reshape(X_new,[self.height, self.width,-1])
       
    def SLIC_Process(self,img,scale=25):
        n_segments_init=self.height*self.width/scale
        print("n_segments_init",n_segments_init)
        myslic=SLIC(img,n_segments=n_segments_init,labels=self.labes, compactness=1,sigma=1, min_size_factor=0.1, max_size_factor=2)

        Q, S, Segments = myslic.get_Q_and_S_and_Segments()
        A=myslic.get_A_my(sigma=1)
        return Q,S,A,Segments
        
    def simple_superpixel(self,scale):
        # curr_labels = self.init_labels
        X = self.data
        Q, S, A, Seg = self.SLIC_Process(X,scale=scale)
        return Q, S, A,Seg
