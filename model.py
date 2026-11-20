import torch
import torch.nn as nn


class CumulativeEmbedding(nn.Module):
    """
    A linear map applied to every frame, summed over time. The thesis (3.6.1)
    calls this a Lie group embedding, but there is no Lie group operation in
    it; for a real one see PathDevelopment (REV-8, PEER_REVIEW EIC-2).
    """
    def __init__(self, input_dim, embed_dim):
        super(CumulativeEmbedding, self).__init__()
        self.fc = nn.Linear(input_dim, embed_dim)

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
        self.lie_embed_joint = CumulativeEmbedding(input_dim=9, embed_dim=16)
        self.dynamic_adj_joint = DynamicAdjacencyMatrix(input_dim=16, num_nodes=42)
        self.residual_joint = ResidualCTRGCNLayer(in_channels=16, out_channels=64, num_nodes=42)
        self.lstm_joint = nn.LSTM(input_size=64 * 42, hidden_size=64, batch_first=True)  # 2688 in features

        # Bone path (20 bones)
        self.lie_embed_bone = CumulativeEmbedding(input_dim=6, embed_dim=16)
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
    the same Z. CumulativeEmbedding (a linear map summed over frames) does not
    have this property. Only the 3 position channels are developed; velocity
    and acceleration change with speed by definition.

    Output (B, T, N, m*m): Z_t flattened, so m=4 matches embed_dim=16.
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


class FixedAdjacency(nn.Module):
    """
    The hand's own skeleton as the adjacency, the same for every sample and
    with nothing to train: (A + I) with each row divided by its sum, so each
    node averages itself and its neighbours (the control for SimilarityAdjacency,
    which learns who is connected from the sample).
    """

    def __init__(self, edges, num_nodes):
        super(FixedAdjacency, self).__init__()
        A = torch.eye(num_nodes)
        for i, j in edges:
            A[i, j] = A[j, i] = 1.0
        self.register_buffer("A", A / A.sum(dim=-1, keepdim=True))

    def forward(self, X):
        return self.A.expand(X.shape[0], -1, -1)  # (B, N, N)


def skeleton_edges():
    """
    (joint edges over the 42 joint nodes, bone edges over the 20 bone nodes).
    Joints 0-20 and 21-41 are the two hand slots, each with the hand's bones
    between its joints; the slots are not joined. Two bones are neighbours
    when they share a joint.
    """
    from pipeline import BONE_CONNECTIONS  # here, not at the top: pipeline needs mediapipe
    joints = [(a + slot, b + slot) for slot in (0, 21) for a, b in BONE_CONNECTIONS]
    bones = [(i, j) for i, (a, b) in enumerate(BONE_CONNECTIONS) for j, (c, d) in enumerate(BONE_CONNECTIONS)
             if i < j and {a, b} & {c, d}]
    return joints, bones


class ResidualGCNLayer(nn.Module):
    """
    Aggregate over the adjacency, 1x1 conv, residual, then BatchNorm + ReLU
    (thesis 3.6.3-3.6.6). form "default" puts the ReLU after the residual sum
    (relu(A.conv(x) + res(x))); form "thesis" follows equation 3.15 as written,
    H = relu(A.conv(x)) + res(x), the residual added after the activation.
    Aggregating before or after the 1x1 conv is the same thing, because every
    row of A sums to 1.
    """

    def __init__(self, in_channels, out_channels, form="default"):
        super(ResidualGCNLayer, self).__init__()
        if form not in ("default", "thesis"):
            raise ValueError(f"form must be 'default' or 'thesis', got {form!r}")
        self.form = form
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, A):
        aggregated = torch.einsum('bnm,btmf->btnf', A, x).permute(0, 3, 1, 2)  # (B, C, T, N)
        residual = self.residual_conv(x.permute(0, 3, 1, 2))
        if self.form == "thesis":
            out = self.relu(self.conv(aggregated)) + residual
        else:
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

    def __init__(self, num_classes, embedding="cumulative", state_dim=0,
                 lstm_input=None, bidir_ends=False, output_gelu=True, use_bones=True, use_gcn=True,
                 adjacency="similarity", gcn_form="default", adj_input="embedded"):
        """
        embedding: "cumulative" (linear map summed over frames; what existing
        checkpoints use) or "development" (PathDevelopment, the speed-invariant
        Lie group embedding the thesis describes).
        state_dim: size of the per-frame hand-state features (wrist height and
        angle with velocity and acceleration: 6) appended to the joint path
        before its LSTM; 0 for none.

        The last three fix implementation details (PEER_REVIEW REV-8, flow
        report M1-M3); their defaults keep the original model, so existing
        checkpoints load unchanged:
        lstm_input: if set, each path's flattened graph output (42x64 or 20x64)
            is projected to this many features before its LSTM (M1). The
            LSTMs hold 99.8% of the original 4.3M parameters, trained on a few
            thousand heavily overlapping windows.
        bidir_ends: take the forward LSTM at the last step and the backward
            LSTM at the first step (M2). At the last step the backward
            direction has read only the last frame.
        output_gelu: apply GELU to the class scores (M3, the original). GELU
            floors scores at about -0.17, so wrong classes cannot be pushed
            down, only the right one up.

        Two ablations, to find which part of the architecture the result
        needs (REV-8); the defaults keep the full model:
        use_bones: False drops the whole bone path (its embedding, graph and
            LSTM); the classifier then reads the joint path alone.
        use_gcn: False drops each path's adjacency and graph layer, so the
            embedding's 16 features per node go straight to the LSTM.
        adjacency: "similarity" (the thesis: learned from each sample's node
            embeddings) or "fixed" (the hand skeleton, nothing learned). Only
            this differs between the two; the graph layer is the same.
        gcn_form: "default" or "thesis", where the graph layer puts its ReLU
            (see ResidualGCNLayer; equation 3.15 of the thesis).
        adj_input: what the learned adjacency is computed from: "embedded"
            (the cumulative embedding's output, 16 features per node, the
            default) or "raw" (the window's own 9 features per node, which is
            what thesis 3.6.2 says; the thesis's 3.6 overview says the
            embedded ones).
        """
        super(StrokeBiLSTMModel, self).__init__()
        if embedding not in self.EMBEDDINGS:
            raise ValueError(f"embedding must be one of {self.EMBEDDINGS}, got {embedding!r}")
        self.embedding, self.state_dim = embedding, state_dim
        self.lstm_input, self.bidir_ends, self.output_gelu = lstm_input, bidir_ends, output_gelu
        self.use_bones, self.use_gcn = use_bones, use_gcn
        if adjacency not in ("similarity", "fixed"):
            raise ValueError(f"adjacency must be 'similarity' or 'fixed', got {adjacency!r}")
        self.adjacency = adjacency
        if adj_input not in ("embedded", "raw"):
            raise ValueError(f"adj_input must be 'embedded' or 'raw', got {adj_input!r}")
        self.gcn_form, self.adj_input = gcn_form, adj_input
        adj_features = 16 if adj_input == "embedded" else 9
        if adjacency == "fixed":
            joint_edges, bone_edges = skeleton_edges()

        def embed():
            return CumulativeEmbedding(input_dim=9, embed_dim=16) if embedding == "cumulative" else PathDevelopment()

        features = 64 if use_gcn else 16  # per node, what reaches the LSTM

        def project(nodes):
            return nn.Linear(features * nodes, lstm_input) if lstm_input else nn.Identity()

        path_size = {nodes: lstm_input or features * nodes for nodes in (42, 20)}

        self.lie_embed_joint = embed()
        if use_gcn:
            self.adj_joint = (FixedAdjacency(joint_edges, 42) if adjacency == "fixed"
                              else SimilarityAdjacency(input_dim=adj_features, embed_dim=32))
            self.gcn_joint = ResidualGCNLayer(in_channels=16, out_channels=64, form=gcn_form)
        self.proj_joint = project(42)
        self.lstm_joint = nn.LSTM(input_size=path_size[42] + state_dim, hidden_size=128, batch_first=True,
                                  bidirectional=True)

        if use_bones:
            self.lie_embed_bone = embed()
            if use_gcn:
                self.adj_bone = (FixedAdjacency(bone_edges, 20) if adjacency == "fixed"
                                 else SimilarityAdjacency(input_dim=adj_features, embed_dim=32))
                self.gcn_bone = ResidualGCNLayer(in_channels=16, out_channels=64, form=gcn_form)
            self.proj_bone = project(20)
            self.lstm_bone = nn.LSTM(input_size=path_size[20], hidden_size=128, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256 * (2 if use_bones else 1), num_classes)
        self.gelu = nn.GELU() if output_gelu else nn.Identity()

    def _summary(self, out):
        """(B, T, 256) Bi-LSTM output -> (B, 256) for the classifier."""
        if self.bidir_ends:
            return torch.cat([out[:, -1, :128], out[:, 0, 128:]], dim=1)
        return out[:, -1]

    def forward(self, x, state=None):
        """x: (B, T, 62, 9). state: (B, T, state_dim), required when state_dim > 0."""
        B, T, _, _ = x.shape

        x_joint = self.lie_embed_joint(x[:, :, :42, :])  # (B, T, 42, 16)
        if self.use_gcn:
            raw = x[:, :, :42, :] if self.adj_input == "raw" else x_joint
            x_joint = self.gcn_joint(x_joint, self.adj_joint(raw))  # (B, T, 42, 64)
        x_joint = self.proj_joint(x_joint.reshape(B, T, -1))
        if self.state_dim:
            x_joint = torch.cat([x_joint, state], dim=-1)
        out_joint, _ = self.lstm_joint(x_joint)  # (B, T, 256)

        summaries = [self._summary(out_joint)]
        if self.use_bones:
            x_bone = self.lie_embed_bone(x[:, :, 42:, :])  # (B, T, 20, 16)
            if self.use_gcn:
                raw = x[:, :, 42:, :] if self.adj_input == "raw" else x_bone
                x_bone = self.gcn_bone(x_bone, self.adj_bone(raw))  # (B, T, 20, 64)
            out_bone, _ = self.lstm_bone(self.proj_bone(x_bone.reshape(B, T, -1)))  # (B, T, 256)
            summaries.append(self._summary(out_bone))

        fused = torch.cat(summaries, dim=1)  # (B, 512), or (B, 256) without the bone path
        return self.gelu(self.fc(self.dropout(fused)))  # (B, num_classes)


class StrokeGRUModel(nn.Module):
    """
    Raw keypoints -> GRU -> last hidden state -> class scores; baselines.py's
    "gru" row, which matched or beat StrokeBiLSTMModel on this data. Takes the
    same packed input (B, T, 62, 9) and keeps only the 42 joint slots'
    positions (channels 0-2); the unused hand's slot is zero. No hand-state input.
    """

    state_dim = 0  # inference.py passes hand-state features only when this is set

    def __init__(self, num_classes, hidden=64):
        super().__init__()
        self.gru = nn.GRU(input_size=42 * 3, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        B, T = x.shape[:2]
        _, h = self.gru(x[:, :, :42, :3].reshape(B, T, -1))
        return self.fc(h[-1])