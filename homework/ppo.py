import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from kart_env import SuperTuxKartEnv  # Custom environment
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = ""

    capture_video: bool = False
    total_timesteps: int = 50000
    learning_rate: float = 1e-4
    num_envs: int = 1
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.0

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def preprocess_observation(obs):
    return obs / 255.0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.observation_space.shape  # (96, 128, 3)
        action_dim = envs.action_space.shape[0]

        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[2], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, obs_shape[2], obs_shape[0], obs_shape[1])
            flattened_size = self.cnn(dummy_input).shape[1]

        self.fc = layer_init(nn.Linear(flattened_size, 512))
        self.relu = nn.ReLU()
        self.actor_mean = layer_init(nn.Linear(512, action_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def get_value(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        x = self.relu(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = self.cnn(x)
        x = self.fc(x)
        x = self.relu(x)
        mean = self.actor_mean(x)
        logstd = self.actor_logstd.expand_as(mean)
        std = torch.exp(logstd)
        if torch.isnan(mean).any() or torch.isnan(std).any():
            print("NaNs detected inside policy network output! Skipping this forward pass.")
            return (
                torch.zeros_like(mean),  # safe action
                torch.zeros(x.shape[0], device=x.device),  # safe log_prob
                torch.zeros(x.shape[0], device=x.device),  # safe entropy
                torch.zeros(1, device=x.device)  # safe value
            )
        dist = Normal(mean, std)

        if action is None:
            action = dist.sample()
        action = torch.clamp(action, -1, 1)

        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(x)
        return action, log_prob, entropy, value


class BatchObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space

    def observation(self, observation):
        return observation


def make_env(track_name):
    def thunk():
        env = SuperTuxKartEnv(track=track_name)
        env = BatchObsWrapper(env)
        return env
    return thunk


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    track_name = "lighthouse"

    if args.num_envs == 1:
        envs = make_env(track_name)()
    else:
        envs = gym.vector.SyncVectorEnv([make_env(track_name) for _ in range(args.num_envs)])

    assert isinstance(envs.action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs, 3, 96, 128)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()

    next_obs, _ = envs.reset(seed=args.seed)
    if isinstance(next_obs, list) or isinstance(next_obs, tuple):
        next_obs = next_obs[0]
    next_obs = torch.tensor(preprocess_observation(next_obs), dtype=torch.float32).permute(1, 2, 0).to(device)
    next_obs = torch.clip(next_obs, 0, 1)
    next_done = torch.zeros(args.num_envs).to(device)
    #print("next_obs shape after reset:", next_obs.shape)

    # Initialize lists to store loss values
    policy_losses = []
    value_losses = []
    entropy_losses = []
    total_losses = []

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs.unsqueeze(0))
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs_, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            # Add this to render:
            if args.num_envs == 1:
                envs.render()

            if isinstance(next_obs_, list) or isinstance(next_obs_, tuple):
                next_obs_ = next_obs_[0]
            next_obs = torch.tensor(preprocess_observation(next_obs_), dtype=torch.float32).permute(1, 2, 0).to(device)
            next_obs = torch.clip(next_obs, 0, 1)

            if isinstance(next_obs_, list) or isinstance(next_obs_, tuple):
                next_obs_ = next_obs_[0]
            next_obs = torch.tensor(preprocess_observation(next_obs_), dtype=torch.float32).permute(1, 2, 0).to(device)
            next_obs = torch.clip(next_obs, 0, 1)
            next_done = torch.tensor(np.logical_or(terminations, truncations), dtype=torch.float32).to(device)
            rewards[step] = torch.tensor(reward).to(device).view(-1)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        with torch.no_grad():
            next_value = agent.get_value(next_obs.unsqueeze(0)).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1, 3, 96, 128))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # NEW: check if this minibatch has NaNs and skip if needed
                if torch.isnan(b_obs[mb_inds]).any() or torch.isnan(b_actions[mb_inds]).any():
                    print(f"Skipping minibatch {start}-{end} due to NaNs")
                    continue

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                # Store the losses
                policy_losses.append(pg_loss.item())
                value_losses.append(v_loss.item())
                entropy_losses.append(entropy_loss.item())
                total_losses.append(loss.item())

            if args.target_kl and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print(f"Iteration {iteration}, SPS: {int(global_step / (time.time() - start_time))}")
        print(f"Total Loss: {loss.item()}")
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()


    plt.figure(figsize=(10, 6))
    plt.plot(total_losses, label="Total Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Losses During PPO Training")
    plt.legend()
    plt.grid()
    plt.ylim(-1, 10)
    plt.show()

    # Smooth the total losses using a Gaussian filter
    smoothed_loss = gaussian_filter1d(total_losses, sigma=2)

    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_loss, label="Smoothed Total Loss", color="red")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Smoothed Losses During PPO Training")
    plt.legend()
    plt.grid()
    plt.ylim(-1, 10)
    plt.show()