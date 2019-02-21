import torch.nn as nn
# from torch.autograd import Variable # Variable is no longer needed from pytorch > 0.4
import torch


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        :param input_size: Height and width of input tensor as (height, width).
        :type input_size: (int, int)
        :param input_dim: Number of channels of input tensor.
        :type input_dim: int
        :param hidden_dim: Number of channels of hidden state.
        :type hidden_dim: int
        :param kernel_size: Size of the convolutional kernel.
        :type kernel_size: (int, int)
        :param bias: Whether or not to add the bias.
        :type bias: bool
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        # padding needs to grantee that the size of the images is the same after the convolution
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2  # this is the same as (kernel_size -1 )/2
        self.bias = bias

        # This Conv LSTM Cell implements the structure shown in:
        # https://jaxenter.com/convolutional-lstm-deeplearning4j-146157.html

        # If we stack hidden state and input we don't have to add the results of the convolutions anymore
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,  # for input, output, forget, and gate-gate
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        # Dimensions of the tensors are (batch_size, channels, heigth, width)
        
        h_cur, c_cur = cur_state  # Get Cell and hidden State. They are 3D tensors + 1D for batch size
        
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        
        combined_conv = self.conv(combined)

        # The actual LSTM computations
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size):
        if torch.cuda.is_available():
            return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width).cuda(),
                    torch.zeros(batch_size, self.hidden_dim, self.height, self.width).cuda())
        else:
            return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width),
                    torch.zeros(batch_size, self.hidden_dim, self.height, self.width))


class ConvLSTM(nn.Module):

    def __init__(self, input_size,
                 input_dim,
                 hidden_dim,
                 kernel_size,
                 num_layers,
                 batch_first=False,
                 bias=True,
                 return_all_layers=True):
        """
        Initialize Multilayer ConvLSTM

        :param input_size: Size of the first input frame(s).
        :type input_size: (int, int)
        :param input_dim: Number of channels of the first input frame(s).
        :param hidden_dim: If list: A list of int giving the number of hidden channels.
                           If int a list will be created containing num_layer times the given int.
        :type hidden_dim: list or int
        :param kernel_size: If list: A list of (int, int) giving the kernel sizes for the Conv LSTM layers.
                            If (int, int) is given a list will be created containing num_layer times the given tuple.
        :type kernel_size: list or (int, int)
        :param num_layers: number of layers
        :param batch_first: If true the input data is expected to be organized as
                            (batch, sequence, channels, height, width)
        :type batch_first: bool
        :param bias: Whether or not to add the bias.
        :param return_all_layers: If True all hidden states and outputs of all layers will be returned in forward
        """

        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        # Create a list of Conv LSTM cells
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        # make sure that if the net is moved to GPU all parameters are moved as well
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        :param input_tensor: 5-D Tensor either of shape (sequence, batch, c, h, w) or (batch, sequence, c, h, w)
        :param hidden_state: The last hidden state to be used as a starting point. If None the hidden state will be set to 0
        :return: last_state_list, layer_output
        """

        if not self.batch_first:
            # (sequence, v, c, h, w) -> (batch, sequence, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Reset state if needed
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]  # get the last hidden state of the current layer
            output_inner = []
            for t in range(seq_len):  # loop over the complete sequence and store the results

                # forward pass through the cell
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)  # store the result

            # The output sequence of dim (batch, sequence, c, h, w) will be the input of the next layer / cell
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        # Note if batch is not first the output should be permute again
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1]
            last_state_list = last_state_list[-1]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
