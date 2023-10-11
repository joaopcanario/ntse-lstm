import math
import torch

from src import EmbeddedModel


def get_indicator(length_tensor, max_length=None):
    lengths_size = length_tensor.size()

    flat_lengths = length_tensor.view(-1, 1)

    if not max_length:
        max_length = length_tensor.max()
    unit_range = torch.arange(max_length)

    flat_range = unit_range.expand(
        flat_lengths.size()[0:1] + unit_range.size())
    flat_indicator = flat_range < flat_lengths

    return flat_indicator.view(lengths_size + (-1, 1))


def get_module_device(model):
    return next(model.parameters()).device


class LSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()

        self.input_size, self.hidden_size = input_size, hidden_size
        self.weights = torch.nn.Linear((input_size + hidden_size),
                                       hidden_size * 4, bias=bias)
        self.cnn = EmbeddedModel.cnn()
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)

        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, X, init_states=None):
        hx, cx = init_states

        X_hx = torch.cat([X, hx], dim=1)
        X_hx = self.weights(X_hx)

        fio_gates, cell = torch.split(X_hx, self.hidden_size * 3, dim=1)
        f, i, o = torch.split(fio_gates, self.hidden_size, dim=1)

        f = self.cnn(f.unsqueeze(1))
        i = self.cnn(i.unsqueeze(1))
        o = self.cnn(o.unsqueeze(1))
        cell = self.cnn(cell.unsqueeze(1))

        cell = torch.tanh(cell)

        f = torch.sigmoid(f)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)

        cy = (f * cx) + (i * cell)
        hy = o * torch.tanh(cy)

        return hy, cy


class LSTMFrame(torch.nn.Module):
    def __init__(self, rnn_cells, batch_first=False, dropout=0, bidirectional=False):
        """
        :param rnn_cells: ex) [(cell_0_f, cell_0_b), (cell_1_f, cell_1_b), ..]
        :param dropout:
        :param bidirectional:
        """
        super().__init__()

        if bidirectional:
            assert all(len(pair) == 2 for pair in rnn_cells)
        elif not any(isinstance(rnn_cells[0], iterable)
                     for iterable in [list, tuple, torch.nn.ModuleList]):
            rnn_cells = tuple((cell,) for cell in rnn_cells)

        self.rnn_cells = torch.nn.ModuleList(torch.nn.ModuleList(pair)
                                             for pair in rnn_cells)
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = len(rnn_cells)

        if dropout > 0 and self.num_layers > 1:
            # dropout is applied to output of each layer except the last layer
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = lambda x: x

        self.batch_first = batch_first

    def align_sequence(self, seq, lengths, shift_right):
        """
        :param seq: (seq_len, batch_size, *)
        """
        multiplier = 1 if shift_right else -1
        example_seqs = torch.split(seq, 1, dim=1)
        max_length = max(lengths)
        shifted_seqs = [example_seq.roll((max_length - length) * multiplier, dims=0)
                        for example_seq, length in zip(example_seqs, lengths)]
        return torch.cat(shifted_seqs, dim=1)

    def forward(self, input, init_state=None):
        """
        :param input: a tensor(s) of shape (seq_len, batch, input_size)
        :param init_state: (h_0, c_0) where the size of both is (num_layers * num_directions, batch, hidden_size)
        :returns: (output, (h_n, c_n))
        - output: (seq_len, batch, num_directions * hidden_size)
        - h_n: (num_layers * num_directions, batch, hidden_size)
        - c_n: (num_layers * num_directions, batch, hidden_size)
        """

        if isinstance(input, torch.nn.utils.rnn.PackedSequence):
            input_packed = True
            # always batch_first=False --> trick to process input regardless of batch_first option
            input, lengths = torch.nn.utils.rnn.pad_packed_sequence(
                input, batch_first=False)
            if max(lengths) == min(lengths):
                uniform_length = True
            else:
                uniform_length = False
            assert max(lengths) == input.size()[0]
        else:
            input_packed = False
            if self.batch_first:
                input = input.transpose(0, 1)
            lengths = [input.size()[0]] * input.size()[1]
            uniform_length = True

        if not uniform_length:
            indicator = get_indicator(torch.tensor(
                lengths, device=get_module_device(self)))

        if init_state is None:
            # init_state with heterogenous hidden_size
            init_hidden = init_cell = [
                torch.zeros(
                    input.size()[1],
                    self.rnn_cells[layer_idx][direction].hidden_size,
                    device=get_module_device(self))
                for layer_idx in range(self.num_layers)
                for direction in range(self.num_directions)]
            init_state = init_hidden, init_cell

        init_hidden, init_cell = init_state

        last_hidden_list = []
        last_cell_list = []

        layer_output = input

        for layer_idx in range(self.num_layers):
            layer_input = layer_output
            if layer_idx != 0:
                layer_input = self.dropout(layer_input)

            direction_output_list = []

            for direction in range(self.num_directions):
                cell = self.rnn_cells[layer_idx][direction]
                state_idx = layer_idx * self.num_directions + direction
                step_state = (init_hidden[state_idx], init_cell[state_idx])

                direction_output = torch.zeros(
                    layer_input.size()[:2] + (cell.hidden_size,),
                    device=get_module_device(self)
                )  # (seq_len, batch_size, hidden_size)
                step_state_list = []

                if direction == 0:
                    step_input_gen = enumerate(layer_input)
                else:
                    step_input_gen = reversed(list(enumerate(
                        layer_input if uniform_length else
                        self.align_sequence(layer_input, lengths, True))))

                for seq_idx, cell_input in step_input_gen:
                    h, c = step_state = cell(cell_input, step_state)

                    direction_output[seq_idx] = h
                    step_state_list.append(step_state)
                if direction == 1 and not uniform_length:
                    direction_output = self.align_sequence(
                        direction_output, lengths, False)

                if uniform_length:
                    direction_last_hidden, direction_last_cell = step_state_list[-1]
                else:
                    direction_last_hidden, direction_last_cell = map(
                        lambda x: torch.stack([
                            x[length - 1][example_id]
                            for example_id, length in enumerate(lengths)
                        ], dim=0),
                        zip(*step_state_list))

                direction_output_list.append(direction_output)
                last_hidden_list.append(direction_last_hidden)
                last_cell_list.append(direction_last_cell)

            if self.num_directions == 2:
                layer_output = torch.stack(direction_output_list, dim=2).view(
                    direction_output_list[0].size()[:2] + (-1,))
            else:
                layer_output = direction_output_list[0]

        output = layer_output
        last_hidden_tensor = torch.stack(last_hidden_list, dim=0)
        last_cell_tensor = torch.stack(last_cell_list, dim=0)

        if not uniform_length:
            # the below one line code cleans out trash values beyond the range of lengths.
            # actually, the code is for debugging, so it can be removed to enhance computing speed slightly.
            output = (
                output.transpose(0, 1) * indicator).transpose(0, 1)

        if input_packed:
            output = torch.nn.utils.rnn.pack_padded_sequence(
                output, lengths, batch_first=self.batch_first)
        elif self.batch_first:
            output = output.transpose(0, 1)

        return output, (last_hidden_tensor, last_cell_tensor)


class CustomLayerLSTM(LSTMFrame):
    def __init__(self, input_size, hidden_size, num_layers=1,
                 batch_first=False, dropout=0, bidirectional=False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

        rnn_cells = tuple(
            tuple(
                LSTMCell(input_size if layer_idx == 0 else hidden_size * (2 if bidirectional else 1),
                         hidden_size)
                for _ in range(2 if bidirectional else 1)
            )

            for layer_idx in range(num_layers)
        )

        super().__init__(rnn_cells=rnn_cells, dropout=dropout,
                         batch_first=batch_first, bidirectional=bidirectional)
