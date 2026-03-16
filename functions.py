from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import tensorflow as tf
# import tensorflow_probability as tfp

def mahalanobis_distance_tf(x, mu, inv_cov):
    """
    使用 TensorFlow 计算 Mahalanobis 距离
    x: 单个像素的特征向量，形状为 (75,)
    mu: 中心值（Huber 均值），形状为 (75,)
    inv_cov: 协方差矩阵的逆，形状为 (75, 75)
    """
    diff = x - mu
    # diff: shape (75,)
    # inv_cov: shape (75, 75)
    distance = tf.sqrt(tf.matmul(tf.matmul(diff[tf.newaxis, :], inv_cov), diff[:, tf.newaxis]))
    return tf.squeeze(distance)  # 返回标量


def compute_mahalanobis_distances_tf(S, center):
    """
    使用 TensorFlow 计算每个像素到中心的 Mahalanobis 距离。
    S: 输入数据，形状为 (n_samples, 75)
    center: 中心点（Huber 均值），形状为 (75,)
    """
    # 标准化 S 和 center
    S = tf.convert_to_tensor(S, dtype=tf.float32)
    center = tf.convert_to_tensor(center, dtype=tf.float32)
    mean = tf.reduce_mean(S, axis=0, keepdims=True)
    std = tf.math.reduce_std(S, axis=0, keepdims=True)
    S = (S - mean) / (std + 1e-8)
    center = (center - mean) / (std + 1e-8)

    # 计算协方差矩阵及其逆
    # cov_matrix = tfp.stats.covariance(S, sample_axis=0)
    mean1 = tf.reduce_mean(S, axis=0)
    # 中心化数据
    centered_data = S - mean1
    # 计算协方差矩阵
    cov_matrix = tf.matmul(centered_data, centered_data, transpose_a=True) / (S.shape[0] - 1)
    epsilon = 1e-8
    cov_matrix += tf.eye(tf.shape(cov_matrix)[0]) * epsilon  # 防止奇异矩阵
    inv_cov_matrix = tf.linalg.pinv(cov_matrix)

    # 使用 tf.map_fn 对每个样本计算距离
    distances = tf.map_fn(lambda x: mahalanobis_distance_tf(x, center, inv_cov_matrix), S)
    return distances

def huber_mean_with_optimized_delta_tf(data, tol=1e-6, max_iter=100):
    """
    计算多维Huber均值，动态优化 delta（使用 TensorFlow）。
    data: 输入数据，形状为 (n_samples, n_features)
    """
    # 初始均值
    mu = tf.reduce_mean(data, axis=0)
    epsilon = 1e-8
    delta = 0.1
    for iteration in range(max_iter):
        diff = data - mu
        distances = tf.norm(diff, axis=1)  # 使用 TensorFlow 的 norm
        # 动态优化 delta
        # delta = optimize_delta(diff, delta)  # 可选的优化方法
        # 计算权重
        weights = tf.where(distances <= delta,
                           1.0,
                           delta / (distances + epsilon))  # 权重计算

        # 更新均值
        weighted_data = weights[:, tf.newaxis] * data  # 权重扩展到数据维度
        new_mu = tf.reduce_sum(weighted_data, axis=0) / (tf.reduce_sum(weights) + epsilon)

        # 判断是否收敛
        if tf.norm(new_mu - mu) < tol:
            break
        # 限制最大更新量
        max_update = 0.01
        update = tf.clip_by_value(new_mu - mu, -max_update, max_update)
        mu = mu + update
    return mu

def normalize(inputs):
    min_value = tf.reduce_min(inputs)  # 计算最小值
    max_value = tf.reduce_max(inputs)  # 计算最大值
    center_new = (inputs - min_value) / (max_value - min_value)
    return center_new


def compute_homogeneity(input_set, num_patch, dimen):
    set_tensor = tf.convert_to_tensor(input_set, dtype=tf.float32)
    huber_means = tf.map_fn(lambda patch: huber_mean_with_optimized_delta_tf(patch), set_tensor)
    center2_reshaped = tf.reshape(huber_means, (-1, 1))
    # 使用 Min-Max 归一化
    center_new = tf.map_fn(lambda patch: normalize(patch), huber_means)

    # distances = tf.map_fn(lambda patch: compute_mahalanobis_distances_tf(patch[0],patch[1]), (set_tensor, center_new))
    center_new = tf.reshape(center_new, (num_patch,1, dimen))
    distances = tf.norm(set_tensor - center_new, axis=2)
    homogeneity_score = tf.map_fn(lambda patch: normalize(patch), distances)
    homogeneity_scores = tf.math.reduce_variance(homogeneity_score, axis=1)
    return homogeneity_scores.numpy()