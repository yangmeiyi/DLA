import os
import torch
import torch.nn as nn
import random
import numpy as np


## strategy 1 ##
class CustomLoss(nn.Module):
    def __init__(self, epoch_factor=0.1, pt=1.0, num_classes=2, epoches=50, weight_strategy_probability=0.5):
        super(CustomLoss, self).__init__()
        self.num_classes = num_classes
        self.epoch_factor = epoch_factor
        self.pt = pt / self.num_classes
        self.epoches = epoches
        self.weight_strategy_probability = weight_strategy_probability

    def forward(self, inputs, targets, epoch, epochs):
        predicted_probs = torch.softmax(inputs, dim=1)
        class_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-class_loss)
        entropy = -torch.sum(predicted_probs * torch.log(predicted_probs), dim=1)
        equal_probabilities = torch.full((self.num_classes, ), 1.0 / self.num_classes)
        self.entropy = -torch.sum(equal_probabilities * torch.log(equal_probabilities))

        # Define a strategy execution based on probability
        self.weight_strategy_probability = self.weight_strategy_probability ** (self.epoches - epoch)
        execute_strategy = random.random() < self.weight_strategy_probability

        # Increase suppression with epoch number
        if execute_strategy:
            if execute_strategy:
                factor = torch.tensor(np.exp(-self.epoch_factor * epoch), dtype=torch.float32)
                weights = torch.where(((pt < self.pt)) & (entropy > 0.8), (1 - pt) * factor, torch.ones_like(pt))
            else:
                weights = torch.ones_like(class_loss)
        else:
            weights = torch.ones_like(class_loss)

        loss = nn.functional.cross_entropy(inputs, targets, reduction='none') * weights
        return loss.mean()

## strategy 2 ##
# class CustomLoss(nn.Module):
#     def __init__(self, epoch_factor=0.1, pt=1.0, num_calasses=2, entropy=1.0, weight_strategy_probability=0.9):
#         super(CustomLoss, self).__init__()
#         self.epoch_factor = epoch_factor
#         self.pt = pt / num_calasses
#         self.entropy = entropy
#         self.weight_strategy_probability = weight_strategy_probability
#
#     def forward(self, inputs, targets, epoch, epochs):
#         predicted_probs = torch.softmax(inputs, dim=1)
#         class_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
#         pt = torch.exp(-class_loss)
#         entropy = -torch.sum(predicted_probs * torch.log(predicted_probs), dim=1)
#
#         # Define a strategy execution based on probability
#         self.weight_strategy_probability = epoch / epochs
#         execute_strategy = random.random() < self.weight_strategy_probability
#
#         # Increase suppression with epoch number
#         if execute_strategy:
#             factor = 1.0 + epoch * 0.5
#             weights = torch.where((pt < self.pt) & (entropy > 0.8), pt ** factor, 1.0)
#
#         else:
#             weights = torch.ones_like(class_loss)
#         loss = nn.functional.cross_entropy(inputs, targets, reduction='none') * weights
#
#         return loss.mean()