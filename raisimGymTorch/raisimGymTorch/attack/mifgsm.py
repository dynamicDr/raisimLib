import copy

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from attack.attack import Attack


class MIFGSM(Attack):
    r"""
    MI-FGSM in the paper 'Boosting Adversarial Attacks with Momentum'
    [https://arxiv.org/abs/1710.06081]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        decay (float): momentum factor. (Default: 1.0)
        steps (int): number of iterations. (Default: 10)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.MIFGSM(model, eps=8/255, steps=10, decay=1.0)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, decay=1.0):
        super().__init__("MIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
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

        momentum = torch.zeros_like(inputs).detach().to(self.device)

        #adv_inputs = inputs.clone().detach() + self.eps * self.last_grad.sign()
        adv_inputs = inputs.clone().detach() + 0.001 * torch.randn_like(inputs)

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

            # print(grad.shape)
            # print(torch.mean(torch.abs(grad), dim=1, keepdim=True))
            grad = grad / torch.mean(torch.abs(grad), dim=1, keepdim=True)
            # grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad

            #print(grad)

            adv_inputs = adv_inputs.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_inputs - inputs, min=-self.eps, max=self.eps)
            adv_inputs = (inputs + delta).detach()
            # adv_inputs = torch.clamp(inputs + delta, min=0, max=1).detach()
            self.last_grad = grad

        return adv_inputs