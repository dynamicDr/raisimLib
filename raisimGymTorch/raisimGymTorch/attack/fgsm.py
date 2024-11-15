import copy

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


from raisimGymTorch.attack.attack import Attack

class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.1)

    Shape:
        - inputs: :math:`(N, D1, D2, ..., Dk)` where `N = number of batches`.
        - output: :math:`(N, D1, D2, ..., Dk)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, eps=0.1)
        >>> adv_inputs = attack(inputs)

    """

    def __init__(self, model, eps=0.1):
        super().__init__("FGSM", model)
        self.eps = eps
        self.supported_mode = ["default", "targeted"]
        self.last_grad = torch.zeros(1, model.obs_dim, dtype=torch.float32).to(self.device) #obs.shape ; eval_obs.shape
        self.gt_model = copy.deepcopy(model)

    def forward(self, inputs, labels=None):
        # TODO
        gt_act = self.gt_model.architecture(inputs)
        labels = gt_act
        '''
        注意inputs(即obs)要与labels（即攻击前的策略网络）所使用的obs相一致！！！
        '''

        inputs = inputs.clone().detach().to(self.device)

        inputs.requires_grad = True
        #last_adv_inputs = inputs + self.eps * self.last_grad.sign()
        adv_inputs = inputs.clone().detach() + 0.001 * torch.randn_like(inputs)

        adv_inputs.requires_grad = True

        # Calculate loss
        dist = self.model.architecture(adv_inputs)
        print(dist)
        actions = dist.rsample()

        cost = torch.nn.functional.mse_loss(actions, labels)

        # Update adversarial inputs
        grad = torch.autograd.grad(
            cost, adv_inputs, retain_graph=False, create_graph=False
        )[0] #这里梯度计算要一直计算到input输入的那个神经网络（输入层）的位置
        #print(grad)
        #print("Gradient shape:", grad.shape)
        adv_inputs = inputs + self.eps * grad.sign()
        adv_inputs = torch.clamp(adv_inputs, min=inputs.min(), max=inputs.max()).detach()
        self.last_grad = grad

        return adv_inputs