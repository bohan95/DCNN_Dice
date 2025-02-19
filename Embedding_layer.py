import torch.nn as nn

class ROIFeatureExtractor(nn.Module):
    def __init__(self, embedding_layer, ROI_EMBEDDING_DIM, hidden_dim):
        super(ROIFeatureExtractor, self).__init__()
        self.embedding = embedding_layer  # 直接使用已有 embedding
        self.mlp = nn.Sequential(
            nn.Linear(ROI_EMBEDDING_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ROI_EMBEDDING_DIM)
        )
        print(f"using ROI with emb: {ROI_EMBEDDING_DIM}")

    def forward(self, roi_input):
        embedded = self.embedding(roi_input)  # (b, 100, embedding_dim)
        features = self.mlp(embedded)  # 让 ROI 通过 MLP 提取更有效的特征
        return features