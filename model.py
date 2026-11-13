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


class PathDevelopment(nn.Module):
    """
    Lie group embedding that the thesis describes (1.4, 2.8.3, 3.6.1): each
    node's position path is "developed" into the rotation group SO(m). Every
    frame-to-frame step of the path is mapped to a skew-symmetric matrix (an
    element of the Lie algebra so(m)) and the steps are multiplied up as
    matrix exponentials:

        Z_t = Z_{t-1} @ expm(M(x_t - x_{t-1})),  Z_0 = I

    M is linear with no bias, so the result depends on the path, not on how
    fast it was traversed: splitting a straight step into two half steps
    gives exactly the same Z, and a curved path played slower gives nearly
    the same Z. LieGroupEmbedding (a linear map summed over frames) does not
    have this property. Only the 3 position channels are developed; velocity
    and acceleration change with speed by definition.

    Output (B, T, N, m*m): Z_t flattened, so m=4 matches lie_dim=16.
    """

    def __init__(self, path_dim=3, group_dim=4):
        super(PathDevelopment, self).__init__()
        self.path_dim, self.group_dim = path_dim, group_dim
        self.fc = nn.Linear(path_dim, group_dim * group_dim, bias=False)

    def forward(self, x):
        B, T, N, _ = x.shape
        m = self.group_dim
        path = x[..., :self.path_dim]
        steps = torch.diff(path, dim=1, prepend=path[:, :1])  # first step is zero
        A = self.fc(steps).view(B, T, N, m, m)
        A = A - A.transpose(-1, -2)  # skew-symmetric: in so(m)
        exps = torch.matrix_exp(A)  # in SO(m)
        Z = exps[:, 0]
        out = [Z]
        for t in range(1, T):
            Z = Z @ exps[:, t]
            out.append(Z)
        return torch.stack(out, dim=1).reshape(B, T, N, m * m)


class SimilarityAdjacency(nn.Module):
    """Sample-specific adjacency from node-embedding similarity (thesis 3.6.2)."""

    def __init__(self, input_dim, embed_dim):
        super(SimilarityAdjacency, self).__init__()
        self.node_fc = nn.Linear(input_dim, embed_dim)

    def forward(self, X):
        H = self.node_fc(X.mean(dim=1))  # (B, N, D): time-averaged, then embedded
        return torch.softmax(H @ H.transpose(-1, -2), dim=-1)  # (B, N, N)


class ResidualGCNLayer(nn.Module):
    """Aggregate over the adjacency, 1x1 conv, residual + ReLU, then BatchNorm + ReLU (thesis 3.6.3-3.6.6)."""

    def __init__(self, in_channels, out_channels):
        super(ResidualGCNLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, A):
        aggregated = torch.einsum('bnm,btmf->btnf', A, x).permute(0, 3, 1, 2)  # (B, C, T, N)
        residual = self.residual_conv(x.permute(0, 3, 1, 2))
        out = self.relu(self.conv(aggregated) + residual)
        out = self.relu(self.bn(out))
        return out.permute(0, 2, 3, 1)  # (B, T, N, C)


class StrokeBiLSTMModel(nn.Module):
    """
    Thesis architecture: joint and bone paths, each cumulative embedding ->
    dynamic adjacency -> residual GCN -> Bi-LSTM (hidden 128), last time step
    of both paths concatenated, Dropout 0.5 -> Linear -> GELU. Returns the
    pre-softmax scores; softmax is applied by the loss / at inference.
    """

    EMBEDDINGS = ("cumulative", "development")

    def __init__(self, num_classes, embedding="cumulative", state_dim=0):
        """
        embedding: "cumulative" (linear map summed over frames; what existing
        checkpoints use) or "development" (PathDevelopment, the speed-invariant
        Lie group embedding the thesis describes).
        state_dim: size of the per-frame hand-state features (wrist height and
        angle with velocity and acceleration: 6) appended to the joint path
        before its LSTM; 0 for none.
        """
        super(StrokeBiLSTMModel, self).__init__()
        if embedding not in self.EMBEDDINGS:
            raise ValueError(f"embedding must be one of {self.EMBEDDINGS}, got {embedding!r}")
        self.embedding, self.state_dim = embedding, state_dim

        def embed():
            return LieGroupEmbedding(input_dim=9, lie_dim=16) if embedding == "cumulative" else PathDevelopment()

        self.lie_embed_joint = embed()
        self.adj_joint = SimilarityAdjacency(input_dim=16, embed_dim=32)
        self.gcn_joint = ResidualGCNLayer(in_channels=16, out_channels=64)
        self.lstm_joint = nn.LSTM(input_size=64 * 42 + state_dim, hidden_size=128, batch_first=True, bidirectional=True)

        self.lie_embed_bone = embed()
        self.adj_bone = SimilarityAdjacency(input_dim=16, embed_dim=32)
        self.gcn_bone = ResidualGCNLayer(in_channels=16, out_channels=64)
        self.lstm_bone = nn.LSTM(input_size=64 * 20, hidden_size=128, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(4 * 128, num_classes)
        self.gelu = nn.GELU()

    def forward(self, x, state=None):
        """x: (B, T, 62, 9). state: (B, T, state_dim), required when state_dim > 0."""
        B, T, _, _ = x.shape

        x_joint = self.lie_embed_joint(x[:, :, :42, :])  # (B, T, 42, 16)
        x_joint = self.gcn_joint(x_joint, self.adj_joint(x_joint))  # (B, T, 42, 64)
        x_joint = x_joint.reshape(B, T, -1)
        if self.state_dim:
            x_joint = torch.cat([x_joint, state], dim=-1)
        out_joint, _ = self.lstm_joint(x_joint)  # (B, T, 256)

        x_bone = self.lie_embed_bone(x[:, :, 42:, :])  # (B, T, 20, 16)
        x_bone = self.gcn_bone(x_bone, self.adj_bone(x_bone))  # (B, T, 20, 64)
        out_bone, _ = self.lstm_bone(x_bone.reshape(B, T, -1))  # (B, T, 256)

        fused = torch.cat([out_joint[:, -1], out_bone[:, -1]], dim=1)  # (B, 512)
        return self.gelu(self.fc(self.dropout(fused)))  # (B, num_classes)
