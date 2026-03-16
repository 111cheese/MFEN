%clear all
clc
I =WHU_Hi_LongKou;
I = double(I);
[h,w,num_bands] = size(I);
I = hyperConvert2d(I);
[p, N] = size(I);
%I = hyperConvert3d(I,h,w,p);
I=I.';

lambda = 0.01;

[A_hat1,E_hat1,iter1] = inexact_alm_rpca(I, lambda);

RPCA = A_hat1;
M2 = hyperConvert3d(RPCA.', h, w, p);
save data_rpcaWHU_LK M2;



