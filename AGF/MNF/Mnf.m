%最大噪声分离
% 参考文献  C-I Change and Q Du,Interference and Noise-Adjusted Pricipal Components Analysis
clear;clc
% readfile;%读入原始数据
%load 92av3c.mat

load  PaviaU.mat
%load  Indian_pines.mat
%load  Indian_pines.mat

corrected=double(paviaU);
[h,w,num_bands] = size(paviaU);
%corrected=double(indian_pines_corrected);
%[h,w,num_bands] = size(indian_pines_corrected);
%corrected=double(salinas_corrected);
%[h,w,num_bands] = size(salinas_corrected);

%转换为二维数据
M=hyperConvert2d(paviaU);
%M=hyperConvert2d(indian_pines_corrected);
%M=hyperConvert2d(salinas_corrected);
%M=hyperConvert2d(data);
[p, N] = size(M);
%计算原始数据的协方差矩阵,调用函数hyperCov
sigmaZ = hyperCov(M);
%转化为三维矩阵以估算噪声协方差矩阵
M = hyperConvert3d(M, h, w, p);

% 估算噪声协方差矩阵
dX = zeros(h-1, w, p);
for i=1:(h-1)
    dX(i, :, :) = M(i, :, :) - M(i+1, :, :);
end
dX = hyperConvert2d(dX);
sigmaN = hyperCov(dX);
%求得噪声协方差矩阵的特征向量并归一化
[U1,S1,E] = svd(sigmaN);
F = E/inv(sqrt(S1));

%对噪声数据进行标准PCA变换
sigmaAdj = F'*sigmaZ*F;
[U2,S2,G] = svd(sigmaAdj);

%得到了MNF的变换矩阵
H = G*F;

% 计算SNR
snr = diag(S2)-1;

% 进行MNF变换后的矩阵
M = H*hyperConvert2d(M);

%存储MNF变换结果
M = hyperConvert3d(M, h, w, p);
save data_mnfsallyj M;
% write_raw('data_mnf', M);

%MNF逆变换
% readfile;%读入MNF变换后的数据 :data_mnf
data_mnf=hyperConvert2d(corrected);
M_2D = F'*G'*data_mnf;%原始图像二维数据
%转换为原始数据的三维形式，并存储
M_restruct = hyperConvert3d(M_2D, h, w, p);
save M_restruct M_restruct;
% write_raw('M_restruct', M_restruct);





