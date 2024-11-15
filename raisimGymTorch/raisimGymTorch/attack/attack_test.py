# Copyright 2023 OmniSafeAI Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from torchsummary import summary
import argparse
import ast
import os
import json
import re
from collections import deque

from fgsm_backup import FGSM
from safepo.common.env import make_sa_mujoco_env, make_ma_mujoco_env, make_ma_multi_goal_env, make_sa_isaac_env
from safepo.common.model import ActorVCritic
from safepo.utils.config import multi_agent_velocity_map, multi_agent_goal_tasks
import numpy as np
import joblib
import torch



eval_dir= "runs/42_lidar/SafetyRacecarGoal0-v0/trpo/seed-000-2024-07-25-10-20-43"
torch.set_num_threads(1)
config_path = eval_dir + '/config.json'
config = json.load(open(config_path, 'r'))

env_id = config['task'] if 'task' in config.keys() else config['env_name']
print(config['task'])
norms = os.listdir(eval_dir)
norms = [norm for norm in norms if norm.endswith('.pkl')]
if len(norms) != 0:
    norms_numbers = [(norm, int(re.search(r'\d+', norm).group())) for norm in norms]
    final_norm_name = max(norms_numbers, key=lambda x: x[1])[0]

model_dir = eval_dir + '/torch_save'
models = os.listdir(model_dir)
models = [model for model in models if model.endswith('.pt')]
model_numbers = [(model, int(re.search(r'\d+', model).group())) for model in models]
final_model_name = max(model_numbers, key=lambda x: x[1])[0]

model_path = model_dir + '/' + final_model_name
if len(norms) != 0:
    norm_path = eval_dir + '/' + final_norm_name
eval_env, obs_space, act_space = make_sa_mujoco_env(num_envs=1, env_id=env_id, seed=None)

hidden_sizes = ast.literal_eval(config['hidden_size'])
model = ActorVCritic(
    obs_dim=obs_space.shape[0],
    act_dim=act_space.shape[0],
    hidden_sizes=hidden_sizes,
)
model.actor.load_state_dict(torch.load(model_path))

summary(model.actor, input_size=(1, 468), device="cpu")
summary(model.reward_critic, input_size=(1, 468), device="cpu")

if len(norms) != 0:
    if os.path.exists(norm_path):
        norm = joblib.load(open(norm_path, 'rb'))['Normalizer']
        print(norm)
        eval_env.obs_rms = norm

eval_rew_deque = deque(maxlen=50)
eval_cost_deque = deque(maxlen=50)
eval_len_deque = deque(maxlen=50)




actor_attacker = FGSM(model.actor, eps=8/255)
critic_attacker = FGSM(model.reward_critic, eps=8/255)




for _ in range(1):
    eval_done = False
    eval_obs, _ = eval_env.reset()
    eval_obs = torch.as_tensor(eval_obs, dtype=torch.float32)
    eval_rew, eval_cost, eval_len = 0.0, 0.0, 0.0
    while not eval_done:
        with torch.no_grad():
            act, _, _, _ = model.step(
                eval_obs, deterministic=True
            )
        eval_obs, reward, cost, terminated, truncated, info = eval_env.step(
            act.detach().squeeze().cpu().numpy()
        )
        # print(eval_obs)
        actor_eval_obs = actor_attacker(eval_obs)
        critic_eval_obs = critic_attacker(eval_obs)
        # print(adv_eval_obs)
        eval_obs = torch.as_tensor(
            eval_obs, dtype=torch.float32
        )
        eval_rew += reward[0]
        eval_cost += cost[0]
        eval_len += 1
        eval_done = terminated[0] or truncated[0]
    eval_rew_deque.append(eval_rew)
    eval_cost_deque.append(eval_cost)
    eval_len_deque.append(eval_len)
    print(eval_rew, eval_len)

