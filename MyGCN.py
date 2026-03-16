import torch
import torch.nn as nn
import torch.nn.functional as F
import AGWConv
from scipy.optimize import fminbound
from torchvision.ops import DeformConv2d


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GWNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, A: torch.Tensor):
        super(GWNLayer, self).__init__()
        self.A = A
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activition = nn.LeakyReLU()
        self.sigma1 = torch.nn.Parameter(torch.tensor([0.1], requires_grad=True))
        # 第一层GCN
        self.GCN_liner_theta_1 = nn.Sequential(nn.Linear(input_dim, 256))
        self.low_dims = nn.Sequential(nn.Linear(input_dim*3, 256))
        self.GCN_liner_out_1 = nn.Sequential(nn.Linear(input_dim, output_dim))
        self.GCN_liner_out_2 = nn.Sequential(nn.Linear(256, output_dim))
        self.MGWN_1 = AGWConv.AGWConv(
            channels=input_dim,
            order=1,
            iterations=3,
            dropout_rate=0.1,
            gcn_activation="relu",  # 或者 "sigmoid", "tanh"
            use_bias=True
        )
        self.nodes_count = self.A.shape[0]
        self.I = torch.eye(self.nodes_count, self.nodes_count, requires_grad=False).to(device)
        self.mask = torch.ceil(self.A * 0.00001)
        self.dropout = nn.Dropout(0.1)


    def laplacian2(self, A: torch.Tensor):
        """
        Compute graph Laplacian from connectivity matrix.
        Parameters:
            A: Adjacency matrix (dense Tensor)
            laplacian_type: Type of Laplacian ('raw' or 'normalized')
        Returns:
            L: Graph Laplacian (sparse Tensor)
        """
        degrees = A.sum(dim=1)  # Compute degree for each node
        diagw = torch.diag(A)  # Extract diagonal entries
        indices = torch.nonzero(A, as_tuple=True)
        ni, nj = indices[0], indices[1]
        w = A[ni, nj]
        ndind = (ni != nj)  # Non-diagonal indices
        ni = ni[ndind]
        nj = nj[ndind]
        w = w[ndind]
        L = torch.diag(degrees - diagw)  # D - A (diagonal part)
        L[ni, nj] = -w  # Off-diagonal entries
        return L

    def sort(self, lamb, U):
        # Sort eigenvalues and eigenvectors in ascending order
        eigenvalue_sorted, idx = torch.sort(lamb)
        eigenvector_sorted = U[:, idx]
        return eigenvalue_sorted, eigenvector_sorted
    def fourier(self, L: torch.Tensor, algo='eigh', k=100):
        # Ensure L is symmetric
        assert torch.allclose(L, L.T, atol=1e-6), "Matrix L must be symmetric."
        if algo == 'eig':
            lamb, U = torch.linalg.eig(L)
            lamb, U = self.sort(lamb.real, U)  # Take real part for eigenvalues
        elif algo == 'eigh':
            lamb, U = torch.linalg.eigh(L)
            lamb, U = self.sort(lamb, U)
        elif algo == 'eigs':
            # For large matrices, you may want to use a method to find the largest/smallest eigenvalues
            eigvals, eigvecs = torch.linalg.eig(L)
            lamb = eigvals.real[:k]
            U = eigvecs[:, :k]
        elif algo == 'eigsh':
            # Similar to 'eigs', but assumes a symmetric matrix
            eigvals, eigvecs = torch.linalg.eigh(L)
            lamb = eigvals[:k]
            U = eigvecs[:, :k]
        return lamb, U

    def weight_wavelet(self, s, lamb, U):
        """
        Compute the wavelet weights based on the given eigenvalues and eigenvectors.
        Parameters
        ----------
        s : float
            Scale parameter for the wavelet.
        lamb : torch.Tensor
            Eigenvalues of the graph Laplacian.
        U : torch.Tensor
            Eigenvectors of the graph Laplacian.
        Returns
        -------
        torch.Tensor
            The computed wavelet weight matrix.
        """
        # Apply wavelet transformation to each eigenvalue using the scale parameter
        lamb = torch.exp(-lamb * s)  # equivalent to math.exp(-lamb[i] * s) for each element in lamb
        Weight = torch.matmul(U, torch.matmul(torch.diag(lamb), U.T))
        return Weight

    def weight_wavelet_inverse(self, s, lamb, U):
        """
        Computes the inverse weight wavelet using PyTorch.
        Parameters:
        s (float): Scale parameter.
        lamb (torch.Tensor): Eigenvalues (1D tensor).
        U (torch.Tensor): Eigenvectors (2D tensor, orthonormal basis).
        """
        # Update eigenvalues using the scale parameter
        lamb = torch.exp(lamb * s)
        # Compute the weight matrix: U * diag(lamb) * U.T
        weight = torch.matmul(U, torch.matmul(torch.diag(lamb), U.T))
        return weight

    # Compute Psi
    def compute_Psi(self, L: torch.Tensor, scales: list):
        """
        Compute wavelet Psi.
        Input:
            L: Laplacian matrix (Tensor)
            scales: List of scales for wavelet
        Returns:
            psi: Wavelet transform
            psi_inv: Inverse wavelet transform
        """
        lamb, U = self.fourier(L)
        psi = []
        psi_inv = []
        for s in scales:
            psi.append(self.weight_wavelet(s, lamb, U))
            psi_inv.append(self.weight_wavelet_inverse(s, lamb, U))
        del U, lamb
        return psi, psi_inv

    def cal_psi(self, A: torch.Tensor,scales, thr=1e-4):
        adj = A.to_dense() if A.is_sparse else A
        L = self.laplacian2(adj)
        # scales = [2 ** i for i in range(N_scales)]
        psi, psi_inv = self.compute_Psi(L, scales)
        # psi, psi_inv = self.approximate_Psi(L, N_scales=3,m=2)
        for i in range(len(psi)):
            psi[i][psi[i] < thr] = 0
            psi_inv[i][psi_inv[i] < thr] = 0
        return psi, psi_inv


    def forward(self, H, high_freq):
        # # 我的方法：图小波卷积
        H2 = self.BN(high_freq)
        H1 = self.BN(H)
        psi, psi_inv = self.cal_psi(self.A, scales=[0.2 ,0.7, 0.9])
        gc1, multi1 = self.MGWN_1(H2, psi, psi_inv, self.A)

        gc2 = self.dropout(gc1)

        gc3, multi2 = self.MGWN_1(gc2, psi, psi_inv, self.A)

        gwn_output = self.Activition(self.GCN_liner_out_1(gc3))

        H_xx1 = self.GCN_liner_theta_1(H1)
        e = torch.sigmoid(torch.matmul(H_xx1, H_xx1.t()))
        zero_vec = -9e15 * torch.ones_like(e)
        A = torch.where(self.mask > 0, e, zero_vec) + self.I
        A = torch.clamp(A, 0.1)
        A = F.softmax(A, dim=1)
        gcn_out = self.Activition(torch.mm(A, self.GCN_liner_out_1(H1)))
        output = gwn_output + gcn_out

        return output

class SSConv(nn.Module):
    '''
    Spectral-Spatial Convolution
    '''
    def __init__(self, in_ch, out_ch,kernel_size=3):
        super(SSConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            groups=out_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.point_conv1 = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.PReLU()
        self.BN=nn.BatchNorm2d(in_ch)
        self.deform = DeformConv2d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.offset_conv = nn.Conv2d(out_ch, 2 * kernel_size * kernel_size, kernel_size=kernel_size, stride=1,
                                     padding=kernel_size//2)
    
    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.Act1(out)
        out1 = self.depth_conv(out)
        out1 = self.Act1(out1)
        out2 = self.point_conv1(out1)
        out2 = self.Act1(out2)
        offset = self.offset_conv(out)
        out3 = self.deform(out, offset)
        out3 = self.Act2(out3)
        out_add = out2 + out3 + out1
        return out_add

class MyGCN(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int, Q: torch.Tensor, A: torch.Tensor, model='normal'):
        super(MyGCN, self).__init__()
        # 类别数,即网络最终输出通道数
        self.class_count = class_count  # 类别数
        # 网络输入数据大小
        self.channel = changel
        self.height = height
        self.width = width
        self.Q = Q
        self.A = A
        self.model=model
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  # 列归一化Q
        self.norm_row_Q = Q / (torch.sum(Q, 1, keepdim=True))  # 行归一化
        
        layers_count=2
        
        # Spectra Transformation Sub-Network
        self.CNN_denoise = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.CNN_denoise.add_module('CNN_denoise_BN'+str(i),nn.BatchNorm2d(self.channel))
                self.CNN_denoise.add_module('CNN_denoise_Conv'+str(i),nn.Conv2d(self.channel, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act'+str(i),nn.LeakyReLU())
            else:
                self.CNN_denoise.add_module('CNN_denoise_BN'+str(i),nn.BatchNorm2d(128),)
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(128, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
        
        # Pixel-level Convolutional Sub-Network
        self.CNN_Branch = nn.Sequential()
        for i in range(layers_count):
            if i<layers_count-1:
                self.CNN_Branch.add_module('CNN_Branch'+str(i),SSConv(128, 128,kernel_size=5))
            else:
                self.CNN_Branch.add_module('CNN_Branch' + str(i), SSConv(128, 64, kernel_size=5))

        self.atten1 = nn.Sequential()
        self.atten1.add_module('Atten1', nn.AdaptiveAvgPool2d(1))
        self.atten2 = nn.Sequential()
        self.atten2.add_module('Atten2', nn.AdaptiveMaxPool2d(1))
        self.attenConv1 = nn.Sequential()
        self.attenConv1.add_module('AttenConv1', nn.Conv1d(128, 1, kernel_size=1))
        self.attenConv1.add_module('AttenConv1_act', nn.PReLU())
        self.attenConv2 = nn.Sequential()
        self.attenConv2.add_module('AttenConv2', nn.Conv1d(1, 1, kernel_size=1))
        self.attenConv2.add_module('AttenConv2_act', nn.PReLU())

        # Superpixel-level Graph Sub-Network
        self.GCN_Branch=nn.Sequential()
        for i in range(layers_count):
            if i<layers_count-1:
                self.GCN_Branch.add_module('GCN_Branch'+str(i),GWNLayer(128, 128, self.A))
            else:
                self.GCN_Branch.add_module('GCN_Branch' + str(i), GWNLayer(128, 64, self.A))

        # Softmax layer
        self.Softmax_linear =nn.Sequential(nn.Linear(128, self.class_count))
    
    def forward(self, x: torch.Tensor):
        '''
        :param x: H*W*C
        :return: probability_map
        '''
        (h, w, c) = x.shape
        
        # 先去除噪声
        noise = self.CNN_denoise(torch.unsqueeze(x.permute([2, 0, 1]), 0))
        noise =torch.squeeze(noise, 0).permute([1, 2, 0])
        clean_x=noise  #直连
        attention1 = self.atten1(clean_x.permute([2, 0, 1]).unsqueeze(0))
        attention2 = self.atten2(clean_x.permute([2, 0, 1]).unsqueeze(0))
        attention_add = attention1 + attention2
        attention_add = attention_add.squeeze(-1).squeeze(-1)
        attention_add = attention_add.unsqueeze(-1)
        atten_conv1 = self.attenConv1(attention_add)
        atten_conv2 = self.attenConv2(atten_conv1)
        atten_all = atten_conv1 + atten_conv2 + attention_add
        atten_all = atten_all.permute([0, 2, 1])
        new_input = clean_x*atten_all

        clean_x_flatten=new_input.reshape([h * w, -1])
        superpixels_flatten = torch.mm(self.norm_col_Q.t(), clean_x_flatten)  # 低频部分
        # high_freq = torch.mm(self.norm_row_Q.t(), clean_x_flatten)
        low_freq = torch.mm(self.norm_col_Q, superpixels_flatten)
        high_freq = clean_x_flatten-low_freq
        high_freq = torch.mm(self.Q.t(), high_freq)
        hx = clean_x
        
        # CNN与GCN分两条支路
        CNN_result = self.CNN_Branch(torch.unsqueeze(hx.permute([2, 0, 1]), 0))# spectral-spatial convolution
        CNN_result = torch.squeeze(CNN_result, 0).permute([1, 2, 0]).reshape([h * w, -1])

        # GCN层 1 转化为超像素 x_flat 乘以 列归一化Q
        H = superpixels_flatten
        # H = torch.mm(self.Q.t(), low_freq)
        if self.model=='normal':
            for i in range(len(self.GCN_Branch)): H = self.GCN_Branch[i](H, high_freq)
        else:
            for i in range(2): H = self.GCN_Branch[i](H, high_freq)
            
            
        GCN_result = torch.matmul(self.Q, H)  # 这里self.norm_row_Q == self.Q
        
        # 两组特征融合(两种融合方式)
        Y = torch.cat([GCN_result,CNN_result],dim=-1)
        Y = self.Softmax_linear(Y)
        Y = F.softmax(Y, -1)
        return Y

if __name__ == "__main__":
    from thop import profile
    #IP数据集
    height, width, bands, class_count = 145, 145, 200, 16
    Q = torch.ones(145*145, 81*4, dtype=torch.float32).to(device)  # 假设 Q 的形状为 (H*W, S)，其中 S 是超像素个数 
    A = torch.ones(81*4, 81*4, dtype=torch.float32).to(device)  # 假设 A 的形状为 (S, S)，其中 S 是超像素个数
    net = MyGCN(height, width, bands, class_count, Q, A, model='smoothed').to(device)
    flops, params = profile(net, inputs=(torch.randn(145,145, 200).to(device),))
    print(f"FLOPs: {flops}, Params: {params}")