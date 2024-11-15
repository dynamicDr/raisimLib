import copy

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from attack.attack import Attack


class TPGD(Attack):
    r"""
    PGD based on KL-Divergence loss in the paper 'Theoretically Principled Trade-off between Robustness and Accuracy'
    [https://arxiv.org/abs/1901.08573]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): strength of the attack or maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.TPGD(model, eps=8/255, alpha=2/255, steps=10)
        >>> adv_images = attack(images)

    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10):
        super().__init__("TPGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.supported_mode = ["default"]

        self.gt_model = copy.deepcopy(model)
        self.model = model
        np.random.seed(0)
        torch.manual_seed(0)

    def forward(self, inputs, labels=None):
        '''
        注意TPGD与其他攻击方法不同，需要输入当前policy以计算KL散度作为损失！！！
        '''
        gt_act, _, _, _ = self.gt_model.step(
            inputs, deterministic=False
        )
        labels = gt_act

        gt_dist = self.gt_model.actor(inputs)

        adv_inputs = (inputs + 0.001 * torch.randn_like(inputs)).detach()
        #adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):

            adv_inputs.requires_grad = True
            adv_dist = self.model.actor(adv_inputs)

            # Calculate loss
            dist = self.model.actor(adv_inputs)
            #dist = self.model(adv_inputs)
            actions = dist.rsample()

            cost1 = torch.nn.functional.mse_loss(actions, labels)

            # compute KL distance between current policy and ground-truth policy as loss (cost)
            cost2 = torch.distributions.kl.kl_divergence(gt_dist, adv_dist).mean()
            #print(adv_dist)
            cost = cost1 + 1 / adv_dist.loc.shape[0] *cost2
            #print(1 / adv_dist.loc.shape[0])

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_inputs, retain_graph=False, create_graph=False
            )[0]

            # print(grad)

            adv_inputs = adv_inputs.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_inputs - inputs, min=-self.eps, max=self.eps)
            adv_inputs = (inputs + delta).detach()
            # adv_inputs = torch.clamp(inputs + delta, min=0, max=1).detach()

        return adv_inputs