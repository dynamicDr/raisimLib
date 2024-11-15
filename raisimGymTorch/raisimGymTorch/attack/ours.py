import copy
import math

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from attack.attack import Attack


def wasserstein_1_distance(dist1, dist2,device="cpu"):
    """
    计算两个多维正态分布之间的 Wasserstein-1 距离。

    参数:
    dist1, dist2: torch.distributions.Normal 对象，假设它们的 loc 和 scale 是多维的

    返回:
    所有维度 Wasserstein-1 距离的平均值
    https://spaces.ac.cn/archives/8512
    https://zh.wikipedia.org/wiki/%E5%A4%9A%E5%85%83%E6%AD%A3%E6%80%81%E5%88%86%E5%B8%83
    """
    mean1, std1 = dist1.loc, dist1.scale
    mean2, std2 = dist2.loc, dist2.scale

    cov1 = torch.diag(std1.pow(2).squeeze())
    cov2 = torch.diag(std1.pow(2).squeeze())

    term1 = torch.sum(torch.abs(mean1 - mean2))
    # term2 = torch.trace(cov1) + torch.trace(cov2) - 2 * torch.trace(torch.sqrt(torch.mm(cov1, cov2))) #在协方差矩阵为对角阵的情况下，term2恒为0

    # return (term1+term2).mean()
    return term1

def jensen_shannon_divergence(p, q,device="cpu"):
    """
    计算两个正态分布之间的 Jensen-Shannon 散度

    参数:
    p, q: torch.distributions.Normal 对象

    返回:
    JS 散度的标量值
    https://blog.csdn.net/zfhsfdhdfajhsr/article/details/127940760
    https://blog.csdn.net/QKK612501/article/details/115370980
    """
    # 创建混合分布 M
    m_mean = (p.loc + q.loc) / 2
    m_scale = torch.sqrt((p.scale ** 2 + q.scale ** 2) / 4)
    m = torch.distributions.Normal(m_mean, m_scale)

    # 计算 KL(P||M) 和 KL(Q||M)
    kl_p_m = torch.distributions.kl.kl_divergence(p, m)
    kl_q_m = torch.distributions.kl.kl_divergence(q, m)

    # 计算 JS 散度
    js = (kl_p_m + kl_q_m) / 2

    return js.mean()


def bhattacharyya_distance(p, q,device="cpu"):
    """
    计算两个正态分布之间的 Bhattacharyya 距离

    参数:
    p, q: torch.distributions.Normal 对象

    返回:
    Bhattacharyya 距离的标量值
    https://blog.csdn.net/bornfree5511/article/details/103811887
    """
    mu1, sigma1 = p.loc, p.scale
    mu2, sigma2 = q.loc, q.scale

    sigma = 0.5 * (sigma1.pow(2) + sigma2.pow(2))

    term1 = 0.25 * (mu1 - mu2).pow(2) / sigma


    term2 = 0.5 * torch.log(
        sigma / torch.sqrt(sigma1.pow(2) * sigma2.pow(2))
    )

    return (term1 + term2).mean()
#
# def bhattacharyya_distance(p, q,device="cpu"):
#     """
#     计算两个正态分布之间的 Bhattacharyya 距离
#
#     参数:
#     p, q: torch.distributions.Normal 对象
#
#     返回:
#     Bhattacharyya 距离的标量值
#     https://blog.csdn.net/bornfree5511/article/details/103811887
#     """
#     mu1, sigma1 = p.loc, p.scale
#     mu2, sigma2 = q.loc, q.scale
#
#     # print('=====')
#     # print(mu1)
#     # print(mu2)
#     term1 = (mu1 - mu2) ** 2 / (4 * (sigma1 ** 2 + sigma2 ** 2))
#     #print(term1)
#     #term2 = 0.25 * torch.log(0.25 * (sigma1 ** 2 / sigma2 ** 2 + sigma2 ** 2 / sigma1 ** 2 + 2))
#
#     return term1.mean()

def kullback_leibler_divergence(dist1, dist2,device="cpu"):
    """
    计算两个一维正态分布之间的 kl。

    参数:
    dist1, dist2: torch.distributions.Normal 对象

    返回:
    kl散度
    """

    return torch.distributions.kl.kl_divergence(dist1, dist2).mean()

#----------------------

def maximum_mean_discrepancy(px, qy, kernel="multiscale", sample_size=20, device="cpu"):
    """
    计算两组N个二维正态分布之间的平均最大平均差异（MMD）。

    Args:
        px: 第一组N个二维正态分布
        qy: 第二组N个二维正态分布
        kernel: 核函数类型，"multiscale" 或 "rbf"
        sample_size: 每个分布的采样数量

    Returns:
        平均MMD值
    """
    # 确定目标设备
    device = device

    N = px.loc.shape[0]  # 分布的数量
    total_mmd = torch.tensor(0.0, device=device)

    for i in range(N):
        x = px.rsample((sample_size,))[..., i, :].to(device)  # 形状: [sample_size, 2]
        y = qy.rsample((sample_size,))[..., i, :].to(device)  # 形状: [sample_size, 2]

        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = (xx.diag().expand_as(xx))
        ry = (yy.diag().expand_as(yy))

        dxx = rx.t() + rx - 2. * xx
        dyy = ry.t() + ry - 2. * yy
        dxy = rx.t() + ry - 2. * zz

        XX = torch.zeros(xx.shape, device=device)
        YY = torch.zeros(xx.shape, device=device)
        XY = torch.zeros(xx.shape, device=device)

        if kernel == "multiscale":
            bandwidth_range = [0.2, 0.5, 0.9, 1.3]
            for a in bandwidth_range:
                a = torch.tensor(a, device=device)
                XX += a**2 * (a**2 + dxx)**-1
                YY += a**2 * (a**2 + dyy)**-1
                XY += a**2 * (a**2 + dxy)**-1

        elif kernel == "rbf":
            bandwidth_range = [10, 15, 20, 50]
            for a in bandwidth_range:
                a = torch.tensor(a, device=device)
                XX += torch.exp(-0.5*dxx/a)
                YY += torch.exp(-0.5*dyy/a)
                XY += torch.exp(-0.5*dxy/a)

        mmd = torch.mean(XX + YY - 2. * XY)
        total_mmd += mmd

    return total_mmd / N

class OURS(Attack):

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, norm = 'wd',device="cpu"):
        super().__init__("TPGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.supported_mode = ["default"]
        self.device = device

        self.last_grad = torch.zeros(1, model.actor.obs_dim, dtype=torch.float32).to(self.device)
        self.gt_model = copy.deepcopy(model)
        self.model = model

        np.random.seed(0)
        torch.manual_seed(0)

        if norm == 'kl':
            self.norm = kullback_leibler_divergence

        elif norm == 'wd':
            self.norm = wasserstein_1_distance

        elif norm == 'js':
            self.norm = jensen_shannon_divergence

        elif norm == 'bd':
            self.norm = bhattacharyya_distance

        elif norm == 'mmd':
            self.norm = maximum_mean_discrepancy


    # def forward(self, inputs, labels=None):
    #     '''
    #     注意TPGD与其他攻击方法不同，需要输入当前policy以计算KL散度作为损失！！！
    #     '''
    #     # print(inputs.shape)
    #     gt_dist = self.gt_model.actor(inputs)
    #     #print(gt_dist)
    #
    #     # adv_inputs = inputs.clone().detach() + self.eps * self.last_grad.sign() + 0.001 * torch.randn_like(inputs)
    #     adv_inputs = (inputs + torch.rand_like(inputs) * self.eps * 2 - self.eps).detach()
    #     #adv_images = torch.clamp(adv_images, min=0, max=1).detach()
    #
    #     for _ in range(self.steps):
    #         adv_inputs.requires_grad = True
    #         adv_dist = self.model.actor(adv_inputs)
    #
    #         # compute KL distance between current policy and ground-truth policy as loss (cost)
    #         #cost = torch.distributions.kl.kl_divergence(gt_dist, adv_dist).mean()
    #         cost = self.norm(gt_dist, adv_dist)
    #         #print(cost)
    #
    #
    #         # Update adversarial images
    #         grad = torch.autograd.grad(
    #             cost, adv_inputs, retain_graph=False, create_graph=False
    #         )[0]
    #
    #         #print(grad)
    #
    #         adv_inputs = adv_inputs.detach() + self.alpha * grad.sign()
    #         delta = torch.clamp(adv_inputs - inputs, min=-self.eps, max=self.eps)
    #         adv_inputs = (inputs + delta).detach()
    #         # adv_inputs = torch.clamp(inputs + delta, min=0, max=1).detach()
    #         self.last_grad = grad
    #
    #     return adv_inputs

    def forward(self, inputs, labels=None):
        '''
        注意TPGD与其他攻击方法不同，需要输入当前policy以计算KL散度作为损失！！！
        '''
        # print(inputs.shape)
        gt_dist = self.gt_model.actor(inputs)
        #print(gt_dist)

        # adv_inputs = inputs.clone().detach() + self.eps * self.last_grad.sign() + 0.001 * torch.randn_like(inputs)
        adv_inputs = (inputs + torch.rand_like(inputs) * self.eps * 2 - self.eps).detach()
        #adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_inputs.requires_grad = True
            adv_dist = self.model.actor(adv_inputs)

            # compute KL distance between current policy and ground-truth policy as loss (cost)
            #cost = torch.distributions.kl.kl_divergence(gt_dist, adv_dist).mean()
            cost1 = bhattacharyya_distance(gt_dist, adv_dist)
            cost2 = kullback_leibler_divergence(gt_dist, adv_dist)
            #print(cost)
            cost = 0.5 * cost1 + cost2
            #cost = cost2

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_inputs, retain_graph=False, create_graph=False
            )[0]

            #print(grad)

            adv_inputs = adv_inputs.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_inputs - inputs, min=-self.eps, max=self.eps)
            adv_inputs = (inputs + delta).detach()
            # adv_inputs = torch.clamp(inputs + delta, min=0, max=1).detach()
            self.last_grad = grad

        return adv_inputs