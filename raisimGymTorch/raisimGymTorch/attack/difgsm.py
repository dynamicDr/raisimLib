import copy

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from attack.attack import Attack


class DIFGSM(Attack):
    r"""
    DI2-FGSM in the paper 'Improving Transferability of Adversarial Examples with Input Diversity'
    [https://arxiv.org/abs/1803.06978]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        decay (float): momentum factor. (Default: 0.0)
        steps (int): number of iterations. (Default: 10)
        resize_rate (float): resize factor used in input diversity. (Default: 0.9)
        diversity_prob (float) : the probability of applying input diversity. (Default: 0.5)
        random_start (bool): using random initialization of delta. (Default: False)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.DIFGSM(model, eps=8/255, alpha=2/255, steps=10, decay=0.0, resize_rate=0.9, diversity_prob=0.5, random_start=False)
        >>> adv_images = attack(images, labels)

    """

    def __init__(
        self,
        model,
        eps=8 / 255,
        alpha=2 / 255,
        steps=10,
        decay=0.0,
        resize_rate=0.9,
        diversity_prob=0.5,
        random_start=False,
    ):
        super().__init__("DIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]

        self.last_grad = torch.zeros(1, model.actor.obs_dim, dtype=torch.float32).to(self.device)
        self.gt_model = copy.deepcopy(model)

        np.random.seed(0)
        torch.manual_seed(0)

    def input_diversity(self, x):
        orig_size = x.shape[-1]
        resize_size = int(orig_size * self.resize_rate)

        if self.resize_rate < 1:
            small_size = resize_size
            large_size = orig_size
        else:
            small_size = orig_size
            large_size = resize_size

        # 随机选择一个新的大小
        rnd_size = torch.randint(low=small_size, high=large_size, size=(1,), dtype=torch.int32)

        # 重新采样
        rescaled = F.interpolate(x.unsqueeze(1), size=rnd_size.item(), mode="linear", align_corners=False).squeeze(1)

        # 填充到原始大小
        if rnd_size < large_size:
            pad_size = large_size - rnd_size
            pad_left = torch.randint(low=0, high=pad_size.item(), size=(1,), dtype=torch.int32)
            pad_right = pad_size - pad_left

            padded = F.pad(
                rescaled,
                [pad_left.item(), pad_right.item()],
                value=0,
            )
        else:
            padded = rescaled

        return padded if torch.rand(1) < self.diversity_prob else x

    def forward(self, inputs, labels=None):
        r"""
        Overridden.
        """

        gt_act, _, _, _ = self.gt_model.step(
            inputs, deterministic=False
        )
        labels = gt_act

        momentum = torch.zeros_like(inputs).detach().to(self.device)

        #adv_inputs = inputs.clone().detach() + self.eps * self.last_grad.sign()
        adv_inputs = inputs.clone().detach() + 0.001 * torch.randn_like(inputs)

        if self.random_start:
            # Starting at a uniformly random point
            adv_inputs = adv_inputs + torch.empty_like(adv_inputs).uniform_(
                -self.eps, self.eps
            )
            #adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):

            adv_inputs.requires_grad = True
            dist = self.model.actor(self.input_diversity(adv_inputs))
            #dist = self.model(adv_inputs)
            actions = dist.rsample()

            # Calculate loss
            cost = torch.nn.functional.mse_loss(actions, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_inputs, retain_graph=False, create_graph=False
            )[0]

            grad = grad / torch.mean(torch.abs(grad), dim=1, keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad

            #print(grad)

            adv_inputs = adv_inputs.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_inputs - inputs, min=-self.eps, max=self.eps)
            adv_inputs = (inputs + delta).detach()
            # adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            self.last_grad = grad

        return adv_inputs