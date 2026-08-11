"""Visual/text embedder used to train OneRestore prompt embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

from cure.constants import EMBEDDER_TYPES, EMBEDDING_DIM


def load_glove_matrix(path: str | Path, vocabulary: Sequence[str]) -> torch.Tensor:
    vectors: dict[str, torch.Tensor] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            word, *values = line.rstrip().split()
            vectors[word] = torch.tensor([float(value) for value in values])

    rows = []
    for label in vocabulary:
        tokens = label.lower().split("_")
        missing = [token for token in tokens if token not in vectors]
        if missing:
            raise KeyError(f"Missing GloVe tokens for {label!r}: {missing}")
        rows.append(torch.stack([vectors[token] for token in tokens]).mean(0))
    matrix = torch.stack(rows)
    if matrix.shape[1] != 300:
        raise ValueError(f"Expected 300-D GloVe vectors, got {matrix.shape[1]}")
    return matrix


class Backbone(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)
        self.block0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.block1 = resnet.layer1
        self.block2 = resnet.layer2
        self.block3 = resnet.layer3
        self.block4 = resnet.layer4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.block4(x)


class CosineClassifier(nn.Module):
    def __init__(self, temperature: float = 0.05) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, image: torch.Tensor, concepts: torch.Tensor) -> torch.Tensor:
        image = F.normalize(image, dim=-1)
        concepts = F.normalize(concepts, dim=-1)
        return image @ concepts.transpose(0, 1) / self.temperature


class EmbeddingClassifier(nn.Module):
    """ResNet-18 classifier with a GloVe-initialized text classifier head.

    Attribute names match ``_embedder_model_epoch150.tar``. The text branch is
    frozen by default, following the original OneRestore training protocol.
    """

    def __init__(
        self,
        glove_path: str | Path,
        type_names: Sequence[str] = EMBEDDER_TYPES,
        *,
        pretrained_backbone: bool = True,
        freeze_text: bool = True,
    ) -> None:
        super().__init__()
        self.type_names = tuple(type_names)
        self.feat_extractor = Backbone(pretrained=pretrained_backbone)
        self.img_embedder = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout2d(0.35),
        )
        self.img_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.img_final = nn.Linear(1024, EMBEDDING_DIM)
        self.classifier = CosineClassifier(temperature=0.05)
        self.embedder = nn.Embedding(len(self.type_names), 300)
        self.embedder.weight.data.copy_(load_glove_matrix(glove_path, self.type_names))
        self.mlp = nn.Sequential(nn.Linear(300, EMBEDDING_DIM), nn.ReLU(inplace=True))
        self.set_text_trainable(not freeze_text)

    def set_text_trainable(self, trainable: bool) -> None:
        for parameter in (*self.embedder.parameters(), *self.mlp.parameters()):
            parameter.requires_grad = trainable

    def text_embeddings(self) -> torch.Tensor:
        indices = torch.arange(len(self.type_names), device=self.embedder.weight.device)
        return self.mlp(self.embedder(indices))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        mean = images.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        images = (images - mean) / std
        features = self.feat_extractor(images)
        features = self.img_embedder(features)
        features = self.img_avg_pool(features).flatten(1)
        return self.classifier(self.img_final(features), self.text_embeddings())
