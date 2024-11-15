import copy

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from attack.attack import Attack


class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
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

        #adv_inputs = inputs.clone().detach() + self.eps * self.last_grad.sign()
        adv_inputs = inputs.clone().detach() + 0.001 * torch.randn_like(inputs)

        if self.random_start:
            # Starting at a uniformly random point
            adv_inputs = adv_inputs + torch.empty_like(adv_inputs).uniform_(
                -self.eps, self.eps
            ).detach()
            #adv_inputs = torch.clamp(adv_inputs, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_inputs.requires_grad = True

            # Calculate loss
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
            #adv_inputs = torch.clamp(inputs + delta, min=0, max=1).detach()
            adv_inputs = inputs + delta
            self.last_grad = grad

        return adv_inputs