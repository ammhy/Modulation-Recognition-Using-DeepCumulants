import torch
import torch.nn as nn
import numpy as np

# Cumulant calculation function
def calculate_cumulants(iq_data):
    """
    Calculate third-order and fourth-order cumulants.

    Parameters:
    iq_data -- The IQ signal data (shape: [batch_size, seq_len, 2]), where 2 corresponds to I and Q.

    Returns:
    cumulants -- A tensor containing third-order and fourth-order cumulants.
    """
    I = iq_data[:, :, 0]
    Q = iq_data[:, :, 1]

    # Third-order cumulant (mean of I*Q)
    third_order = torch.mean(I * Q, dim=1)

    # Fourth-order cumulant (mean of I*Q*I)
    fourth_order = torch.mean(I * Q * I, dim=1)

    return third_order, fourth_order

# Residual Block Definition
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 如果需要匹配维度，使用捷径连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


# 加入 Transformer 和 LSTM 的 ResNet
class ResNet1DWithTransformer(nn.Module):
    def __init__(self, block, layers, num_classes, input_size, num_heads=4, num_transformer_layers=2, d_model=256,
                 lstm_hidden_size=128, lstm_num_layers=1, dropout_rate=0.3):
        super(ResNet1DWithTransformer, self).__init__()
        self.in_channels = 64
        # 初始卷积层
        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        # 残差层
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, d_model, layers[3], stride=2)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # LSTM 层
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers,
                            batch_first=True, dropout=dropout_rate)

        # Dropout 层用于正则化
        self.dropout = nn.Dropout(p=dropout_rate)

        # 全连接层用于最终分类
        self.fc = nn.Linear(lstm_hidden_size + 2, num_classes)  # 融合 IQ 信号 和 高阶累积量特征

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, iq_data):
        """
        前向传播函数，接收 IQ 信号作为输入。
        """
        # 计算高阶累积量
        third_order, fourth_order = calculate_cumulants(iq_data)

        # IQ 信号通过 ResNet 分支
        x = iq_data.permute(0, 2, 1)  # (batch_size, channels, seq_len)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, d_model)

        # 通过 Transformer 编码器
        x = self.transformer_encoder(x)

        # LSTM 层
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # 取最后一个时间步的输出

        # 融合 IQ 特征 和 高阶累积量特征
        cumulant_data = torch.cat((third_order.unsqueeze(1), fourth_order.unsqueeze(1)), dim=1)  # 连接高阶累积量特征
        combined_features = torch.cat((x, cumulant_data), dim=1)
        combined_features = self.dropout(combined_features)

        # 通过全连接层进行最终分类
        output = self.fc(combined_features)
        return output


def ResNet18WithTransformer(num_classes, input_size):
    return ResNet1DWithTransformer(ResidualBlock1D, [2, 2, 2, 2], num_classes=num_classes, input_size=input_size)