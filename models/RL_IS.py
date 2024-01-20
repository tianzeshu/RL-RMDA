import torch
from torch import nn


class policy_selector(nn.Module):
    def __init__(self, input_size):
        super(policy_selector, self).__init__()
        self.affine1 = nn.Linear(input_size, 1)
        self.affine2 = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x, score):
        x = self.affine1(x)
        x = torch.cat([x, score], dim=-1)
        action_scores = self.affine2(x)
        return self.sigmoid(action_scores)


def optimize_selector(_ARGS, sl_model, sl_optimizer, sl_criterion, x_representations, score_list, y_select, rewards):
    sl_model.train()
    sl_optimizer.zero_grad()
    score_list = torch.FloatTensor(score_list).to(_ARGS.device)
    x_representations = torch.stack(x_representations, dim=0).to(_ARGS.device)
    x_representations = x_representations.squeeze()
    y_select = torch.FloatTensor(y_select).to(_ARGS.device)
    y_preds = sl_model(x_representations, score_list)
    y_preds = y_preds.squeeze()
    neg_log_prob = sl_criterion(y_preds, y_select)
    rewards = torch.FloatTensor(rewards).to(_ARGS.device)
    policy_loss = torch.sum(neg_log_prob * rewards)

    lambda1, lambda2 = 0.003, 0.003
    all_linear1_params = torch.cat([x.view(-1) for x in sl_model.affine1.parameters()])
    all_linear2_params = torch.cat([x.view(-1) for x in sl_model.affine1.parameters()])
    l1_regularization = lambda1 * torch.norm(all_linear1_params, 1)
    l2_regularization = lambda2 * torch.norm(all_linear2_params, 2)
    policy_loss += l1_regularization + l2_regularization
    policy_loss.backward()
    sl_optimizer.step()
