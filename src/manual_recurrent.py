import math

import torch
from torch import nn


class _ManualRecurrentBase(nn.Module):
    """手写循环层共用的输入、状态和多层处理工具。"""

    gate_count = 1

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        batch_first=False,
        dropout=0.0,
    ):
        super().__init__()
        if input_size <= 0 or hidden_size <= 0 or num_layers <= 0:
            raise ValueError(
                "input_size、hidden_size 和 num_layers 必须大于 0"
            )
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout 必须位于 [0, 1] 区间")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout

        self.weight_ih = nn.ParameterList()
        self.weight_hh = nn.ParameterList()
        self.bias_ih = nn.ParameterList()
        self.bias_hh = nn.ParameterList()

        gate_size = self.gate_count * hidden_size
        for layer_index in range(num_layers):
            layer_input_size = (
                input_size if layer_index == 0 else hidden_size
            )
            self.weight_ih.append(
                nn.Parameter(torch.empty(gate_size, layer_input_size))
            )
            self.weight_hh.append(
                nn.Parameter(torch.empty(gate_size, hidden_size))
            )
            self.bias_ih.append(nn.Parameter(torch.empty(gate_size)))
            self.bias_hh.append(nn.Parameter(torch.empty(gate_size)))

        self.inter_layer_dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        """采用与 PyTorch 循环层相近的均匀分布初始化。"""
        bound = 1.0 / math.sqrt(self.hidden_size)
        for parameter in self.parameters():
            if parameter.requires_grad:
                nn.init.uniform_(parameter, -bound, bound)

    def _prepare_inputs(self, inputs):
        if inputs.dim() != 3:
            raise ValueError(
                "循环层输入必须是三维 Tensor，"
                "形状为 [batch, sequence, feature]"
            )
        if inputs.size(-1) != self.input_size:
            raise ValueError(
                f"输入特征维度应为 {self.input_size}，"
                f"实际为 {inputs.size(-1)}"
            )

        if not self.batch_first:
            inputs = inputs.transpose(0, 1)
        if inputs.size(1) == 0:
            raise ValueError("sequence_length 必须大于 0")
        return inputs

    def _prepare_hidden(self, hidden, inputs):
        expected_shape = (
            self.num_layers,
            inputs.size(0),
            self.hidden_size,
        )
        if hidden is None:
            return inputs.new_zeros(expected_shape)
        if tuple(hidden.shape) != expected_shape:
            raise ValueError(
                f"隐藏状态形状应为 {expected_shape}，"
                f"实际为 {tuple(hidden.shape)}"
            )
        return hidden

    def _finish_layer(self, layer_output, layer_index):
        if layer_index < self.num_layers - 1 and self.dropout > 0:
            return self.inter_layer_dropout(layer_output)
        return layer_output

    def _restore_output_layout(self, outputs):
        if not self.batch_first:
            outputs = outputs.transpose(0, 1)
        return outputs


class ManualRNN(_ManualRecurrentBase):
    """使用 tanh 显式实现的基础 RNN。"""

    gate_count = 1

    def forward(self, inputs, state=None):
        inputs = self._prepare_inputs(inputs)
        hidden = self._prepare_hidden(state, inputs)

        layer_input = inputs
        final_hidden = []

        for layer_index in range(self.num_layers):
            hidden_t = hidden[layer_index]
            time_outputs = []

            for time_index in range(layer_input.size(1)):
                input_t = layer_input[:, time_index, :]
                input_projection = (
                    input_t @ self.weight_ih[layer_index].T
                    + self.bias_ih[layer_index]
                )
                hidden_projection = (
                    hidden_t @ self.weight_hh[layer_index].T
                    + self.bias_hh[layer_index]
                )

                # h_t = tanh(W_ih x_t + b_ih + W_hh h_(t-1) + b_hh)
                hidden_t = torch.tanh(
                    input_projection + hidden_projection
                )
                time_outputs.append(hidden_t)

            layer_output = torch.stack(time_outputs, dim=1)
            final_hidden.append(hidden_t)
            layer_input = self._finish_layer(
                layer_output,
                layer_index,
            )

        outputs = self._restore_output_layout(layer_input)
        hidden_n = torch.stack(final_hidden, dim=0)
        return outputs, hidden_n


class ManualGRU(_ManualRecurrentBase):
    """显式实现 reset、update 和 candidate 三组 GRU 门。"""

    gate_count = 3

    def forward(self, inputs, state=None):
        inputs = self._prepare_inputs(inputs)
        hidden = self._prepare_hidden(state, inputs)

        layer_input = inputs
        final_hidden = []

        for layer_index in range(self.num_layers):
            hidden_t = hidden[layer_index]
            time_outputs = []

            for time_index in range(layer_input.size(1)):
                input_t = layer_input[:, time_index, :]
                input_projection = (
                    input_t @ self.weight_ih[layer_index].T
                    + self.bias_ih[layer_index]
                )
                hidden_projection = (
                    hidden_t @ self.weight_hh[layer_index].T
                    + self.bias_hh[layer_index]
                )
                input_r, input_z, input_n = input_projection.chunk(
                    3,
                    dim=-1,
                )
                (
                    hidden_r,
                    hidden_z,
                    hidden_candidate,
                ) = hidden_projection.chunk(3, dim=-1)

                # r_t 决定遗忘多少旧状态，z_t 决定保留多少旧状态。
                reset_gate = torch.sigmoid(input_r + hidden_r)
                update_gate = torch.sigmoid(input_z + hidden_z)
                candidate = torch.tanh(
                    input_n + reset_gate * hidden_candidate
                )
                hidden_t = (
                    (1.0 - update_gate) * candidate
                    + update_gate * hidden_t
                )
                time_outputs.append(hidden_t)

            layer_output = torch.stack(time_outputs, dim=1)
            final_hidden.append(hidden_t)
            layer_input = self._finish_layer(
                layer_output,
                layer_index,
            )

        outputs = self._restore_output_layout(layer_input)
        hidden_n = torch.stack(final_hidden, dim=0)
        return outputs, hidden_n


class ManualLSTM(_ManualRecurrentBase):
    """显式实现 input、forget、candidate 和 output 四组 LSTM 门。"""

    gate_count = 4

    def forward(self, inputs, state=None):
        inputs = self._prepare_inputs(inputs)
        if state is None:
            hidden = self._prepare_hidden(None, inputs)
            cell = self._prepare_hidden(None, inputs)
        else:
            if not isinstance(state, tuple) or len(state) != 2:
                raise ValueError("LSTM 状态必须是 (hidden, cell) 元组")
            hidden = self._prepare_hidden(state[0], inputs)
            cell = self._prepare_hidden(state[1], inputs)

        layer_input = inputs
        final_hidden = []
        final_cell = []

        for layer_index in range(self.num_layers):
            hidden_t = hidden[layer_index]
            cell_t = cell[layer_index]
            time_outputs = []

            for time_index in range(layer_input.size(1)):
                input_t = layer_input[:, time_index, :]
                gates = (
                    input_t @ self.weight_ih[layer_index].T
                    + self.bias_ih[layer_index]
                    + hidden_t @ self.weight_hh[layer_index].T
                    + self.bias_hh[layer_index]
                )
                input_gate, forget_gate, candidate, output_gate = (
                    gates.chunk(4, dim=-1)
                )

                input_gate = torch.sigmoid(input_gate)
                forget_gate = torch.sigmoid(forget_gate)
                candidate = torch.tanh(candidate)
                output_gate = torch.sigmoid(output_gate)

                cell_t = (
                    forget_gate * cell_t
                    + input_gate * candidate
                )
                hidden_t = output_gate * torch.tanh(cell_t)
                time_outputs.append(hidden_t)

            layer_output = torch.stack(time_outputs, dim=1)
            final_hidden.append(hidden_t)
            final_cell.append(cell_t)
            layer_input = self._finish_layer(
                layer_output,
                layer_index,
            )

        outputs = self._restore_output_layout(layer_input)
        hidden_n = torch.stack(final_hidden, dim=0)
        cell_n = torch.stack(final_cell, dim=0)
        return outputs, (hidden_n, cell_n)
