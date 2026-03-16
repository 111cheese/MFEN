import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AGWConv(nn.Module):
    def __init__(
            self,
            channels,
            order=1,
            iterations=3,
            share_weights=False,
            gcn_activation="relu",
            dropout_rate=0.1,
            activation=None,
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            **kwargs
    ):
        super(AGWConv, self).__init__()
        self.channels = channels
        self.iterations = iterations
        self.order = order
        self.share_weights = share_weights
        self.gcn_activation = gcn_activation
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias

        # Define the dropout layer
        self.dropout = nn.Dropout(p=self.dropout_rate)

        # Initialize weights
        self.kernels = []
        for k in range(self.order):
            kernel_stack = []
            current_shape = channels
            for i in range(self.iterations):
                kernel_stack.append(self.create_weights(current_shape))
                current_shape = channels
                if self.share_weights and i == 1:
                    break
            self.kernels.append(kernel_stack)

    def create_weights(self, input_dim):
        """Creates weight parameters for a single convolutional step."""
        kernel_1 = nn.Parameter(torch.Tensor(input_dim, self.channels))
        kernel_2 = nn.Parameter(torch.Tensor(input_dim, self.channels))
        if self.use_bias:
            bias = nn.Parameter(torch.Tensor(self.channels))
        else:
            bias = None

        # Initialize weights
        nn.init.xavier_uniform_(kernel_1)
        nn.init.xavier_uniform_(kernel_2)
        if self.use_bias:
            nn.init.zeros_(bias)

        return kernel_1, kernel_2, bias

    def forward(self, x, phsi, phsiInv, a_in):
        output = []
        output1 = []
        for k in range(self.order):
            output_k = x
            for i in range(self.iterations): #k
                output_k = self.gcs(output_k, x, phsi[k], phsiInv[k], a_in, k, i)
                output.append(output_k)
                output1.append(output_k)

        output = torch.stack(output, dim=-1)
        output = torch.mean(output, dim=-1)

        if self.gcn_activation == "relu":
            output = F.relu(output)
        elif self.gcn_activation == "sigmoid":
            output = torch.sigmoid(output)
        elif self.gcn_activation == "tanh":
            output = torch.tanh(output)
        elif self.gcn_activation == "softmax":
            output = torch.softmax(output, dim=1)

        return output, output1

    def gcs(self, x, x_skip, phsi, phsiInv, a_in, stack, iteration):
        """
        Creates a graph convolutional layer with a skip connection.
        """
        itr = 1 if self.share_weights and iteration >= 1 else iteration
        kernel_1, kernel_2, bias = self.kernels[stack][itr]

        # Adjusting phsi with factor f (as done in TensorFlow)
        f = 1  # This can be adjusted based on specific logic
        phsi = phsi * f

        # Perform graph convolution
        output = torch.matmul(x, kernel_1.to(device))
        output = torch.matmul(phsiInv, output)
        output = torch.matmul(phsi, output)

        skip = torch.matmul(x_skip, kernel_2.to(device))
        skip = self.dropout(skip)
        output += skip

        if bias is not None:
            output += bias.to(device)

        return output


