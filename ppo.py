import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.nn as nn
import imageio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from tqdm import tqdm
import cv2
import humanoid_bench
from stable_baselines3.common.vec_env import SubprocVecEnv
import os


def ortho_mlp_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


def ortho_gru_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'weight_ih' in name:
            nn.init.orthogonal_(param, gain=gain)
        elif 'weight_hh' in name:
            nn.init.orthogonal_(param, gain=gain)
        elif 'bias' in name:
            nn.init.zeros_(param)


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, ortho_init=True):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, out_dim)
        self.act = nn.Tanh()
        if ortho_init == True:
            ortho_mlp_init(self.fc1)
            ortho_mlp_init(self.fc2)
            ortho_mlp_init(self.fc3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        return self.fc3(x)


class GaussMLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, ortho_init=True):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, out_dim)
        self.log_std = nn.Parameter(torch.zeros(out_dim))
        self.act = nn.Tanh()
        if ortho_init == True:
            ortho_mlp_init(self.fc1)
            ortho_mlp_init(self.fc2)
            ortho_mlp_init(self.fc3, gain=0.01)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        mu = self.fc3(x)
        return mu, self.log_std * torch.ones_like(mu)

    def logp(self, mu, log_std, action):
        dis = torch.distributions.Normal(mu, log_std.exp())
        logp = dis.log_prob(action)
        return logp

    def action_prob(self, mu, log_std):
        dis = torch.distributions.Normal(mu, log_std.exp())
        action = dis.rsample()
        log_prob = dis.log_prob(action)
        return action, log_prob


class BetaMLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, ortho_init=True):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, out_dim)
        self.fc4 = nn.Linear(hid_dim, out_dim)
        self.act = nn.Tanh()
        if ortho_init == True:
            ortho_mlp_init(self.fc1)
            ortho_mlp_init(self.fc2)
            ortho_mlp_init(self.fc3, gain=0.01)
            ortho_mlp_init(self.fc4, gain=0.01)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        alpha = (self.fc3(x).exp() + 1.0).log() + 1.0
        beta = (self.fc4(x).exp() + 1.0).log() + 1.0
        return alpha, beta

    def logp(self, alpha, beta, action):
        dis = torch.distributions.Beta(alpha, beta)
        logp = dis.log_prob((action + 1) / 2.0) / 2.0
        return logp

    def action_prob(self, alpha, beta):
        dis = torch.distributions.Beta(alpha, beta)
        action = dis.rsample()
        action = torch.clamp(action, 0.0001, 0.9999)
        log_prob = dis.log_prob(action)
        return action * 2 - 1, log_prob / 2.0


class Network(nn.Module):

    def __init__(self, env_dim,  hid_dim=512, act_dim=61, num_layers=2, ema=0.9, range=8, ortho_init=True):
        super().__init__()
        self.env_dim = env_dim
        self.act_dim = act_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.range = range
        self.ema = ema

        self.input_obs = MLP(env_dim, hid_dim, hid_dim, ortho_init)
        self.input_act = MLP(act_dim, hid_dim, hid_dim, ortho_init)
        self.value = MLP(hid_dim, hid_dim, 1)

        self.output_obs = MLP(hid_dim,  hid_dim, env_dim, ortho_init)
        self.output_act = BetaMLP(hid_dim, hid_dim, act_dim, ortho_init)

        self.GRU = nn.GRU(hid_dim, hidden_size=hid_dim,
                          num_layers=num_layers, batch_first=False)
        if ortho_init == True:
            ortho_gru_init(self.GRU)

    def forward(self, states, actions,  rollout=False, use_mean_action=False):
        # print(states.shape, actions.shape)
        if rollout == False:
            states = states.transpose(0, 1)
            actions = actions.transpose(0, 1)
            answer_action = actions[-1, :, :].clone()
            actions = actions[:-1, :, :].clone()

        assert states.shape[0] == self.range + 1 and actions.shape[0] == self.range, "Length incorrect."

        states = self.input_obs(states)
        actions = self.input_act(actions)
        # Translated to (self.range,Batch,self.hid_dim) shape

        mix = torch.zeros(
            2 * self.range+1, states.shape[1], self.hid_dim, device=states.device)

        mix[0::2, :, :] = states
        mix[1::2, :, :] = actions
        mix = self.GRU(mix)[0]

        if rollout == False:
            state_out_pre = mix[1::2, :, :].clone()
            # state_mu, state_log_std = self.output_obs(state_out_pre)
            # state_out, _ = self.output_obs.action_prob(state_mu, state_log_std)
            state_out = self.output_obs(state_out_pre)
            state_out = state_out.transpose(0, 1)
        else:
            state_out = None

        final = mix[-1, :, :].clone()
        value = self.value(final)

        action_out_pre = mix[-1, :, :].clone()
        alpha, beta = self.output_act(action_out_pre)

        if rollout == False:
            action_log_prob = self.output_act.logp(alpha, beta, answer_action)
            return action_log_prob, value, state_out
        else:
            # for i in range(self.range):
            #     action_out_pre[i+1] = self.ema * \
            #         action_out_pre[i+1] + action_out_pre[i] * (1-self.ema)
            # action_ema = action_out_pre[-1]
            # alpha, beta = self.output_act(action_ema)

            if use_mean_action == False:
                action_out, action_log_prob = self.output_act.action_prob(alpha, beta)
            else:
                action_out = (alpha/(alpha + beta)).clamp(min=0.0001, max=0.9999) * 2 - 1
                action_log_prob = self.output_act.logp(alpha, beta, action_out)

            return action_out, action_log_prob, value


class TrajectoryCollector:

    def __init__(self, init_state, obs_num, act_num, range, reward_scale):
        self.obs_num = obs_num
        self.act_num = act_num
        self.range = range
        self.reward_scale = reward_scale
        self.reset(init_state)

    def reset(self, init_state):
        self.count = 0
        self.s = [torch.zeros(self.obs_num) for _ in range(self.range+1)]
        self.s[-1] = init_state
        self.a = [torch.zeros(self.act_num) for _ in range(self.range)]
        self.r = []
        self.v = []
        self.logp = []
        self.reward_sum = 0
        self.rmean = torch.zeros(1)
        self.rstd = torch.ones(1)

    def length(self):
        return len(self.r)

    def sum_reward(self):
        return self.reward_sum

    def for_sampling(self):
        s_final = torch.stack(self.s[-self.range-1:], dim=0)
        a_final = torch.stack(self.a[-self.range:], dim=0)
        return s_final, a_final

    def append(self, s, r, v, logp, a):
        self.s.append(s)
        self.reward_sum += r.item()
        self.v.append(v)
        self.logp.append(logp)
        self.a.append(a)
        self.count += 1

        # num = self.length()
        # if num > 0:
        #     r = r + discount * self.r[-1]
        # new_mean = self.rmean + \
        #     (r - self.rmean) / (num + 1)
        # if num > 0:
        #     self.rstd = (((num - 1) * self.rstd.pow(2) + (
        #         r - self.rmean) * (r - new_mean)) / num).sqrt()
        # self.rmean = new_mean

        self.r.append((self.reward_scale * r).clamp(min=-10, max=10))

    def to_dataset(self, gae=0.95, discount=0.99):
        adv = [torch.zeros(1) for _ in range(self.count)]

        adv[-1] = self.r[-1] - self.v[-1]

        for i in range(self.count-1):
            adv[-i-2] = self.r[-i-2] + discount * self.v[-i-1] - self.v[-i-2]

        scale = gae * discount
        for i in range(self.count-1):
            adv[-i-2] += adv[-i-1] * scale

        # value = [adv[i]+self.v[i] for i in range(len(adv))]

        # value = [self.r[i] for i in range(len(adv))]
        # for i in range(self.count-1):
        #     value[-i-2] += value[-i-1] * discount

        # if self.count > 1:
        #     v = [self.r[i] for i in range(len(self.r))]
        #     for i in range(self.count-1):
        #         v[-i-2] += v[-i-1] * discount
        #     std = torch.stack(v, dim=0).std()
        #     for i in range(self.count):
        #         self.r[i] = self.r[i] / (std + 1e-8)

        value = self.r
        for i in range(self.count - 1):
            value[-i-2] += self.v[-i-1] * discount

        s_windows = []
        a_windows = []

        for i in range(self.count):

            a_windows.append(torch.stack(self.a[i:i+self.range+1], dim=0))
            s_windows.append(torch.stack(self.s[i:i+self.range+1], dim=0))

        adv_tensor = torch.stack(adv, dim=0)
        logp_tensor = torch.stack(self.logp, dim=0)
        a_tensor = torch.stack(a_windows, dim=1)
        v_tensor = torch.stack(value, dim=0)
        s_tensor = torch.stack(s_windows, dim=1)

        return s_tensor, a_tensor, logp_tensor, adv_tensor, v_tensor


class PPO_MODEL:

    def __init__(self, env: SubprocVecEnv, load_path=None):
        self.name = env_name
        self.device = "cuda:"+str(device)
        self.env = env
        self.env_num = self.env.num_envs
        self.act_num = self.env.action_space.shape[0]
        self.obs_num = self.env.observation_space.shape[0]
        self.obs_mean = torch.zeros(self.obs_num)
        self.obs_std = torch.ones(self.obs_num)
        self.obs_count = 0
        self.range = 8
        self.dim = 512
        self.num_layers = 2

        self.steps = 1000
        self.batch_size = 500
        self.train_epochs = 10
        self.total_steps = 10000000
        self.total_rounds = int((self.total_steps - 1) / (self.steps * self.env_num)) + 1
        self.now_round = 0

        assert self.env_num * self.steps % self.batch_size == 0, "Batchsize is incorrect"
        self.batch_per_round = int(self.env_num * self.steps * self.train_epochs / self.batch_size)

        self.reward_scale = 0.25
        self.init_lr = 4e-5
        self.final_lr = 1e-5
        self.gamma = 0.98
        self.gae = 0.95
        self.clip = 0.2
        self.entropy_scale = 0.05
        self.v_loss_scale = 0.3
        self.state_loss_scale = 0.01
        self.max_grad_norm = 0.5
        self.collectors = []
        self.s_data = []
        self.adv_data = []
        self.logp_data = []
        self.a_data = []
        self.v_data = []

        self.network = Network(self.obs_num, self.dim, self.act_num, self.num_layers, 0.9, self.range)

        if load_path != None:
            all = torch.load(load_path,map_location=lambda storage, loc: storage.cuda(0))
            self.obs_mean = all['obs_mean'].cpu()
            self.obs_std = all['obs_std'].cpu()
            self.obs_count = all['obs_count']
            self.network.load_state_dict(all['network'])
            
        self.network = self.network.to(self.device)
        # self.dp_network = nn.DataParallel(
        #     self.network, device_ids=[1, 2])

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.init_lr, eps=1e-6)

        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
        #                                                             self.total_rounds*self.batch_per_round,
        #                                                             eta_min=self.final_lr)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, 1.0, (self.final_lr/self.init_lr), total_iters=self.total_rounds*self.batch_per_round)

    def train(self):
        tqdmLoader = trange(
            self.train_epochs, desc=f"Training (Begin At Step {(self.now_round - 1) * self.batch_per_round})")

        adv_mean = torch.mean(self.adv_data)
        adv_std = torch.std(self.adv_data)
        for _ in tqdmLoader:
            indices = torch.randperm(self.env_num*self.steps)
            s = self.s_data[indices]
            a = self.a_data[indices]
            v = self.v_data[indices]
            adv = self.adv_data[indices]
            logp = self.logp_data[indices]

            for i in range(0, self.env_num*self.steps, self.batch_size):
                s_input = s[i:i+self.batch_size, :].to(self.device)
                a_input = a[i:i+self.batch_size, :].to(self.device)
                v_input = v[i:i+self.batch_size, :].to(self.device)
                adv_input = adv[i:i+self.batch_size].to(self.device)
                adv_input = (adv_input - adv_mean) / (adv_std + 1e-8)
                logp_input = logp[i:i+self.batch_size].to(self.device)
                # print(s_input.shape, a_input.shape, v_input.shape,
                #      adv_input.shape, logp_input.shape)

                logp_out, v_out, s_out = self.network.forward(
                    s_input, a_input)
                # print(a_out.shape, logp_out.shape, v_out.shape, s_out.shape)

                value_loss = self.v_loss_scale * (v_input - v_out).pow(2).mean()
                state_loss = self.state_loss_scale * (s_input[:, 1:, :] - s_input[:, :-1,:] - s_out).pow(2).mean()
                coef = (logp_out - logp_input).exp()

                # print(logp_out.exp().mean(), logp_input.exp().mean())
                entropy_loss = self.entropy_scale * (coef * logp_out).mean()
                ppo_loss1 = (adv_input * coef)
                ppo_loss2 = (adv_input * torch.clip(coef, 1-self.clip, 1+self.clip))

                value_show = (v_input - v_out).abs().mean()

                ppo_loss = - torch.minimum(ppo_loss1, ppo_loss2).mean()
                loss = value_loss + state_loss + ppo_loss + entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()

                tqdmLoader.set_postfix(lr=f"{self.optimizer.param_groups[0]['lr']:.6f}",
                                       coef=coef.mean().item(),
                                       value_L1_error=value_show.item(),
                                       value_loss=value_loss.item(), state=state_loss.item(),
                                       ppo=ppo_loss.item(), entropy=entropy_loss.item())

    def sample(self):

        with torch.no_grad():

            init_state = torch.tensor(self.env.reset()[0], dtype=torch.float32)
            new_mean = self.obs_mean + (init_state - self.obs_mean) / (self.obs_count + 1)
            if self.obs_count > 0:
                self.obs_std = (((self.obs_count - 1) * self.obs_std.pow(2) + (
                    init_state - self.obs_mean) * (init_state - new_mean)) / self.obs_count).sqrt()
            self.obs_mean = new_mean
            self.obs_count += 1
            init_state = (init_state - self.obs_mean) / (self.obs_std + 1e-8)

            self.collectors = [TrajectoryCollector(init_state, self.obs_num,
                                                   self.act_num, self.range,
                                                   self.reward_scale) for _ in range(self.env_num)]
            self.s_data = []
            self.a_data = []
            self.v_data = []
            self.logp_data = []
            self.adv_data = []

            full_trajectory_num = 0
            full_trajectory_length = 0
            full_reward = 0
            all_reward = 0
            tqdmLoader = trange(
                self.steps, desc=f'Sampling Trajectories(Begin At Sampling Steps {self.now_round * self.env_num * self.steps})')
            tqdmLoader.set_postfix(average_trajectory_length=0,
                                   average_trajectory_reward=0,
                                   average_step_reward=0)

            for i in tqdmLoader:
                s = []
                a = []
                for j in range(self.env_num):
                    s_tensor, a_tensor = self.collectors[j].for_sampling()
                    s.append(s_tensor)
                    a.append(a_tensor)

                s_input = torch.stack(s, dim=1).to(self.device)
                a_input = torch.stack(a, dim=1).to(self.device)
                # print(s_input.shape, self.obs_mean.shape, self.obs_std.shape)

                action, logp, value = self.network.forward(
                    s_input, a_input, rollout=True)
                action = action.cpu()
                logp = logp.cpu()
                value = value.cpu()

                next_state, reward, done, info = self.env.step(action)

                for j in range(self.env_num):
                    # print(next_state[j].shape)
                    obs = torch.tensor(next_state[j], dtype=torch.float32)
                    new_mean = self.obs_mean + (obs - self.obs_mean) / (self.obs_count + 1)
                    self.obs_std = (((self.obs_count - 1) * self.obs_std.pow(2) + (
                        obs - self.obs_mean) * (obs - new_mean)) / self.obs_count).sqrt()
                    self.obs_mean = new_mean
                    self.obs_count += 1
                    obs = (obs - self.obs_mean) / (self.obs_std + 1e-8)

                    self.collectors[j].append(
                        obs, torch.tensor(reward[j], dtype=torch.float32).unsqueeze(0),
                        value[j], logp[j], action[j])

                    if done[j] == True or i == self.steps-1:
                        now_reward = self.collectors[j].sum_reward()
                        all_reward += now_reward

                        if done[j] == True:
                            full_trajectory_num += 1
                            full_trajectory_length += self.collectors[j].length()
                            full_reward += now_reward
                            tqdmLoader.set_postfix(avg_obs_mean=self.obs_mean.abs().mean().item(),
                                                   avg_obs_std=self.obs_std.abs().mean().item(),
                                                   avg_trajectory_length=full_trajectory_length/full_trajectory_num,
                                                   avg_trajectory_reward=full_reward/full_trajectory_num,
                                                   avg_step_reward=all_reward/(i+1)/self.env_num)

                        s_tensor, a_tensor, logp_tensor, adv_tensor, v_tensor = \
                            self.collectors[j].to_dataset(gae=self.gae, discount=self.gamma)

                        self.s_data.append(s_tensor)
                        self.a_data.append(a_tensor)
                        self.logp_data.append(logp_tensor)
                        self.adv_data.append(adv_tensor)
                        self.v_data.append(v_tensor)

                        self.collectors[j].reset(init_state)

            self.s_data = torch.cat(self.s_data, dim=1).transpose(0, 1)
            self.adv_data = torch.cat(self.adv_data, dim=0)
            self.a_data = torch.cat(self.a_data, dim=1).transpose(0, 1)
            self.logp_data = torch.cat(self.logp_data, dim=0)
            self.v_data = torch.cat(self.v_data, dim=0)
        self.now_round += 1

    def gen_video(self, name=None, use_mean_action=True):

        if name == None:
            dir = './' + self.name + '_videos/'
            os.makedirs(dir, exist_ok=True)
            dir = dir + str(self.now_round)+".mp4"
        else:
            dir = name

        print('--- rendering video ---')
        env = gym.make(env_name, render_mode="rgb_array")
        fps = env.metadata['render_fps']

        init_state = torch.tensor(env.reset()[0], dtype=torch.float32)
        init_state = (init_state - self.obs_mean) / (self.obs_std + 1e-8)
        collector = TrajectoryCollector(
            init_state, self.obs_num, self.act_num, self.range, self.reward_scale)

        with torch.no_grad():
            pbar = tqdm(dynamic_ncols=True, desc="Generating " + dir,
                        leave=False, position=0)
            done = False
            video = []

            while done == False:
                s_tensor, a_tensor = collector.for_sampling()

                s_tensor = s_tensor.unsqueeze(1).to(self.device)
                a_tensor = a_tensor.unsqueeze(1).to(self.device)

                action, action_logp, value = self.network(
                    s_tensor.to(self.device), a_tensor.to(self.device),
                    rollout=True, use_mean_action=use_mean_action)

                a = action.cpu().view(-1)
                next_state, reward, done, _, info = env.step(a)
                s = torch.tensor(next_state, dtype=torch.float32).cpu()
                s = (s - self.obs_mean) / (self.obs_std + 1e-8)
                r = torch.tensor(reward, dtype=torch.float32).cpu()
                v = value.cpu()
                logp = action_logp.cpu()

                collector.append(s, r, v, logp, a)
                video.append(env.render())
                pbar.set_postfix(steps=collector.length(),
                                 sum_reward=collector.sum_reward())

                if collector.length() >= 1000:
                    break

        with imageio.get_writer(dir, fps=fps) as writer:
            for frame in video:
                writer.append_data(frame.astype(np.uint8))

        env.close()
        print('\n'+'--- finish rendering---')

    def evaluate(self, use_mean_action=True):

        with torch.no_grad():
            init_state = torch.tensor(self.env.reset()[0], dtype=torch.float32)
            init_state = (init_state - self.obs_mean) / (self.obs_std + 1e-8)
            self.collectors = [TrajectoryCollector(init_state, self.obs_num,
                                                   self.act_num, self.range,
                                                   self.reward_scale) for _ in range(self.env_num)]
            full_trajectory_num = 0
            full_trajectory_length = 0
            full_reward = 0
            all_reward = 0
            tqdmLoader = trange(self.steps, desc=f'Round {self.now_round} Evaluation')
            tqdmLoader.set_postfix(
                average_trajectory_length=0, average_trajectory_reward=0, average_step_reward=0)
            
            for i in tqdmLoader:
                s = []
                a = []
                for j in range(self.env_num):
                    s_tensor, a_tensor = self.collectors[j].for_sampling()
                    s.append(s_tensor)
                    a.append(a_tensor)

                s_input = torch.stack(s, dim=1).to(self.device)
                a_input = torch.stack(a, dim=1).to(self.device)
                action, logp, value = self.network.forward(s_input, a_input, rollout=True, use_mean_action=use_mean_action)

                action = action.cpu()
                logp = logp.cpu()
                value = value.cpu()
                next_state, reward, done, info = self.env.step(action)
                for j in range(self.env_num):
                    obs = torch.tensor(next_state[j], dtype=torch.float32)
                    obs = (obs - self.obs_mean) / (self.obs_std + 1e-8)
                    self.collectors[j].append(
                        obs,torch.tensor(reward[j], dtype=torch.float32).unsqueeze(0),
                        value[j], logp[j], action[j])
                    
                    if done[j] == True or i == self.steps-1:
                        now_reward = self.collectors[j].sum_reward()
                        all_reward += now_reward
                        if done[j] == True:
                            full_trajectory_num += 1
                            full_trajectory_length += self.collectors[j].length()
                            full_reward += now_reward
                            tqdmLoader.set_postfix(avg_obs_mean=self.obs_mean.abs().mean().item(),
                                                   avg_obs_std=self.obs_std.abs().mean().item(),
                                                   avg_trajectory_length=full_trajectory_length/full_trajectory_num,
                                                   avg_trajectory_reward=full_reward/full_trajectory_num,
                                                   avg_step_reward=all_reward/(i+1)/self.env_num)
                        self.collectors[j].reset(init_state)

    def learn(self):
        dir = './' + self.name+'/'
        self.evaluate()
        self.gen_video()
        for _ in range(self.total_rounds):
            self.sample()
            self.train()
            if _ % 10 == 0:
                self.evaluate()
                self.gen_video()
            os.makedirs(dir, exist_ok=True)
            torch.save({'obs_mean': self.obs_mean,
                        'obs_std': self.obs_std,
                        'obs_count': self.obs_count,
                        'network': self.network.state_dict()},
                       dir+str(self.now_round)+'.pt')
        self.gen_video()


def make_env(rank, seed=0):
    def _init():
        env = gym.make(env_name, render_mode="rgb_array")
        env.seed(seed + rank)
        return env
    return _init


task = 3
device = 0
# os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MUJOCO_GL"] = "egl"
if task == 0:
    env_name = "h1hand-walk-v0"
elif task == 1:
    env_name = "h1hand-truck-v0"
elif task == 2:
    env_name = "h1hand-bookshelf_simple-v0"
elif task == 3:
    env_name = "h1hand-sit_hard-v0"
elif task == 4:
    env_name = "h1hand-pole-v0"
elif task == 5:
    env_name = "h1hand-powerlift-v0"
env_num = 32

if __name__ == "__main__":
    # for training
    env = SubprocVecEnv([make_env(i) for i in range(env_num)])     
    PPO_MODEL(env).learn()

    # Generated Videos are in ./videos
    # env_name = "h1hand-walk-v0"
    # env = SubprocVecEnv([make_env(i) for i in range(1)])
    # model = PPO_MODEL(env, './checkpoints/walk.pt')
    # model.gen_video('./videos/walk.mp4')

    # env_name = "h1hand-sit_hard-v0"
    # env = SubprocVecEnv([make_env(i) for i in range(1)])
    # model = PPO_MODEL(env, './checkpoints/sit_hard.pt')
    # model.gen_video('./videos/sit_hard.mp4')

    # env_name = "h1hand-truck-v0"
    # env = SubprocVecEnv([make_env(i) for i in range(1)])
    # model = PPO_MODEL(env, './checkpoints/truck.pt')
    # model.gen_video('./videos/truck.mp4')

    # env_name = "h1hand-bookshelf_simple-v0"
    # env = SubprocVecEnv([make_env(i) for i in range(1)])
    # model = PPO_MODEL(env, './checkpoints/bookshelf_simple.pt')
    # model.gen_video('./videos/bookshelf-simple.mp4')

    exit(0)
