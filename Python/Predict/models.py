import torch.nn as nn
class RegressionNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RegressionNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64, bias=False)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32, bias=False)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.fc5(x)
        return x

class JointLinearModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.selector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1)  # 生成特征权重
        )
        self.predictor = nn.Linear(input_dim, 3)  # 回归任务示例

    def forward(self, x):
        weights = self.selector(x)
        selected_features = x * weights  # 加权特征
        return self.predictor(selected_features), weights

class JointAttnModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.selector = nn.MultiheadAttention(16, num_heads)
        self.predictor = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.projection(x)
        x = x.reshape(4,x.shape[0],-1)
        attn_output, attn_weights = self.selector(x, x, x)
        # feature_importance = attn_weights.mean(dim=1).squeeze()
        attn_output = attn_output.reshape(x.shape[1],-1)
        output = self.predictor(attn_output)
        return output