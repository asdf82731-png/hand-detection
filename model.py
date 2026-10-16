import torch
import torch.nn as nn


class LieGroupEmbedding(nn.Module):
    def __init__(self, input_dim, lie_dim):
        super(LieGroupEmbedding, self).__init__()
        self.fc = nn.Linear(input_dim, lie_dim)

    def forward(self, x):
        batch, time, nodes, features = x.size()
        x = x.reshape(batch * time * nodes, features)  # Safe reshape
        xi = self.fc(x)
        xi = xi.view(batch, time, nodes, -1)
        return torch.cumsum(xi, dim=1)


class DynamicAdjacencyMatrix(nn.Module):
    def __init__(self, input_dim, num_nodes):
        super(DynamicAdjacencyMatrix, self).__init__()
        self.num_nodes = num_nodes
        self.node_fc = nn.Linear(input_dim, num_nodes)

    def forward(self, X):
        # X: (B, T, N, F) -> average over time only, keep each node's identity
        X = X.mean(dim=1)  # (B, N, F)

        A = self.node_fc(X)  # (B, N, N)
        A = (A + A.transpose(-1, -2)) / 2  # Symmetrize
        return torch.softmax(A, dim=-1)


class ResidualCTRGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super(ResidualCTRGCNLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x.clone()
        x = x.permute(0, 3, 1, 2)  # (B, C, T, N)
        x = self.conv(x)
        x = self.bn(x)
        residual = residual.permute(0, 3, 1, 2)
        residual = self.residual_conv(residual)
        out = self.relu(x + residual)  # (B, C, T, N)
        return out.permute(0, 2, 3, 1)  # back to (B, T, N, C)


class MultiMotionDevLSTMModel(nn.Module):
    def __init__(self, num_classes):
        super(MultiMotionDevLSTMModel, self).__init__()

        # Joint path (42 joints)
        self.lie_embed_joint = LieGroupEmbedding(input_dim=9, lie_dim=16)
        self.dynamic_adj_joint = DynamicAdjacencyMatrix(input_dim=16, num_nodes=42)
        self.residual_joint = ResidualCTRGCNLayer(in_channels=16, out_channels=64, num_nodes=42)
        self.lstm_joint = nn.LSTM(input_size=64 * 42, hidden_size=64, batch_first=True)  # 2688 in features

        # Bone path (20 bones)
        self.lie_embed_bone = LieGroupEmbedding(input_dim=6, lie_dim=16)
        self.dynamic_adj_bone = DynamicAdjacencyMatrix(input_dim=16, num_nodes=20)
        self.residual_bone = ResidualCTRGCNLayer(in_channels=16, out_channels=64, num_nodes=20)
        self.lstm_bone = nn.LSTM(input_size=64 * 20, hidden_size=64, batch_first=True)  # 1280 in features

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        B, T, N, _ = x.shape

        # === JOINT PATH ===
        x_joint = self.lie_embed_joint(x[:, :, :42, :9])  # (B,T,42,16)
        A_joint = self.dynamic_adj_joint(x_joint)  # (B,42,42)
        x_joint = torch.einsum('bnm,btmf->btnf', A_joint, x_joint)  # (B,T,42,16)
        x_joint = self.residual_joint(x_joint)  # (B,T,42,64)
        x_joint = x_joint.reshape(B, T, -1)  # (B,T,2688)

        # === BONE PATH ===
        x_bone = self.lie_embed_bone(x[:, :, 42:, :6])  # (B,T,20,16)
        A_bone = self.dynamic_adj_bone(x_bone)  # (B,20,20)
        x_bone = torch.einsum('bnm,btmf->btnf', A_bone, x_bone)  # (B,T,20,16)
        x_bone = self.residual_bone(x_bone)  # (B,T,20,64)
        x_bone = x_bone.reshape(B, T, -1)  # (B,T,1280)

        # === LSTM PROCESSING ===
        _, (h_joint, _) = self.lstm_joint(x_joint)  # (1,B,64)
        _, (h_bone, _) = self.lstm_bone(x_bone)  # (1,B,64)

        # === FINAL CLASSIFICATION ===
        out = torch.cat([h_joint[-1], h_bone[-1]], dim=1)  # (B,128)
        return self.fc(out)  # (B,num_classes)
