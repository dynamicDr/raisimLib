import copy

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from attack.attack import Attack


class RFGSM(Attack):
    r"""
    R+FGSM in the paper 'Ensemble Adversarial Training : Attacks and Defences'
    [https://arxiv.org/abs/1705.07204]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): strength of the attack or maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.RFGSM(model, eps=8/255, alpha=2/255, steps=10)
        >>> adv_images = attack(images, labels)
    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10):
        super().__init__("RFGSM", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.supported_mode = ["default", "targeted"]

        self.last_grad = torch.zeros(1, model.actor.obs_dim, dtype=torch.float32).to(self.device)
        self.gt_model = copy.deepcopy(model)

        np.random.seed(0)
        torch.manual_seed(0)

    def forward(self, inputs, labels=None):

        gt_act, _, _, _ = self.gt_model.step(
            inputs, deterministic=False
        )
        labels = gt_act

        adv_inputs = inputs + self.alpha * torch.randn_like(inputs).sign().detach()

        for _ in range(self.steps):

            adv_inputs.requires_grad = True
            dist = self.model.actor(adv_inputs)
            #dist = self.model(adv_inputs)
            actions = dist.rsample()

            cost = torch.nn.functional.mse_loss(actions, labels)


            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_inputs, retain_graph=False, create_graph=False
            )[0]

            #print(grad)

            adv_inputs = adv_inputs.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_inputs - inputs, min=-self.eps, max=self.eps)
            '''
            原版code中，将adv_inputs的图像像素值都限制在 [0, 1] 范围内，不确定在rl中是否也需要把adv_inputs限制在obs的范围内
            adv_inputs = torch.clamp(inputs + delta, min=0, max=1).detach()
            '''
            adv_inputs = (inputs + delta).detach()

        return adv_inputs