# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
from project.kart_env_sac import SuperTuxKartEnv

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    track_name: str = "lighthouse"
    """the track that the model will train on"""
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""


# Convert SAC's continuous output to dictionary actions
def split_action(action_tensor):
    action = {
        'steer': action_tensor[0].item(),
        'acceleration': action_tensor[1].item(),
        'brake': action_tensor[2] > 0.5,
        'nitro': action_tensor[3] > 0.5,
        'drift': action_tensor[4] > 0.5
    }
    return action


class DictToTensorWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.original_action_space = env.action_space
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(5,), dtype=np.float32)

    def step(self, action):
        action_dict = {
            'steer': action[0],
            'acceleration': action[1],
            'brake': action[2] > 0.5,
            'nitro': action[3] > 0.5,
            'drift': action[4] > 0.5
        }
        return super().step(action_dict)


class TransformFinalObservationWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if "final_observation" in info and info["final_observation"] is not None:
            info["final_observation"] = info["final_observation"].transpose(2, 0, 1).astype(np.float32) / 255.0
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        if "final_observation" in info and info["final_observation"] is not None:
            info["final_observation"] = info["final_observation"].transpose(2, 0, 1).astype(np.float32) / 255.0
        return obs, info


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = SuperTuxKartEnv(track_name=args.track_name)
        
        # First transform observations to CHW format
        env = gym.wrappers.TransformObservation(
            env,
            lambda obs: obs.transpose(2, 0, 1).astype(np.float32) / 255.0,
            gym.spaces.Box(
                low=0, 
                high=1.0, 
                shape=(3, 96, 128),
                dtype=np.float32
            )
        )
        
        # Add wrapper to transform final_observation
        env = TransformFinalObservationWrapper(env)
        
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", 
                episode_trigger=lambda x: x % 100 == 0)
        
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = DictToTensorWrapper(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 96, 128)
            cnn_output_size = self.cnn(dummy).shape[1]

        self.fc1 = nn.Linear(cnn_output_size + 5, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = self.cnn(x)
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(-1)


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 96, 128)
            cnn_output_size = self.cnn(dummy).shape[1]
            
        self.fc_mean_cont = nn.Linear(cnn_output_size, 2)
        self.fc_logstd_cont = nn.Linear(cnn_output_size, 2)
        self.fc_discrete = nn.Linear(cnn_output_size, 3)
        
        # Manual scaling for continuous actions
        self.register_buffer("action_scale", torch.tensor([1.0, 0.5]))
        self.register_buffer("action_bias", torch.tensor([0.0, 0.5]))

    def forward(self, x):
        features = self.cnn(x)
        mean_cont = self.fc_mean_cont(features)
        log_std_cont = torch.tanh(self.fc_logstd_cont(features))
        log_std_cont = LOG_STD_MIN + 0.5*(LOG_STD_MAX - LOG_STD_MIN)*(log_std_cont + 1)
        logits_discrete = self.fc_discrete(features)
        return (mean_cont, log_std_cont), logits_discrete

    def get_action(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(self.action_scale.device)
        (mean_cont, log_std_cont), logits_discrete = self(x)

        # Continuous actions
        std_cont = log_std_cont.exp()
        normal = torch.distributions.Normal(mean_cont, std_cont)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        steer = y_t[..., 0] * self.action_scale[0] + self.action_bias[0]
        accel = y_t[..., 1] * self.action_scale[1] + self.action_bias[1]

        # Discrete actions
        discrete_dist = torch.distributions.Bernoulli(logits=logits_discrete)
        discrete_actions = discrete_dist.sample()

        # Combine actions
        action = torch.cat([
            steer.unsqueeze(-1),
            accel.unsqueeze(-1),
            discrete_actions
        ], dim=-1)

        # Log probs
        log_prob_cont = normal.log_prob(x_t) - torch.log(
            self.action_scale * (1 - y_t.pow(2)) + 1e-6
        )
        log_prob_discrete = discrete_dist.log_prob(discrete_actions)

        return action, log_prob_cont.sum(-1) + log_prob_discrete.sum(-1), None



if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -0.5 * (2 * (1 + np.log(2*np.pi)) + 3 * np.log(2))  # Keep this line
        log_alpha = torch.zeros(1, requires_grad=True, device=device)  # Keep existing initialization
        alpha = log_alpha.exp().item()  # Keep existing alpha setup
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        gym.spaces.Box(low=0, high=1, shape=(3, 96, 128)),  # Normalized observations
        gym.spaces.Box(low=-1, high=1, shape=(5,)),  # Flattened actions
        device,
        handle_timeout_termination=False
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # Action logic
        if global_step < args.learning_starts:
            # Random actions matching your action space structure
            actions = np.array([[
                np.random.uniform(-1, 1),  # steer
                np.random.uniform(0, 1),    # acceleration
                np.random.randint(2),       # brake
                np.random.randint(2),       # nitro
                np.random.randint(2)        # drift
            ] for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(device)
                action_dict, _, _ = actor.get_action(obs_tensor)

        # Environment step
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx in range(envs.num_envs):
            if truncations[idx] or terminations[idx]:
                if "final_observation" in infos:
                    real_next_obs[idx] = infos["final_observation"][idx]
                else:  # Fallback for environments without final_observation
                    real_next_obs[idx] = infos.get("final_observation", next_obs)[idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            
            # Convert array actions to dict for Q networks
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                
                # Get Q-values for next state
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                
                # Take min Q-value and subtract entropy term
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                
                # Calculate target Q-value (shape: [batch_size])
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * min_qf_next_target.flatten()
                
                # Clamp Q-values
                next_q_value = next_q_value.clamp(-1/(1-args.gamma), 1/(1-args.gamma))

            # Q function update
            current_q1 = qf1(data.observations, data.actions)
            current_q2 = qf2(data.observations, data.actions)
            qf1_loss = F.mse_loss(current_q1.view(-1), next_q_value)
            qf2_loss = F.mse_loss(current_q2.view(-1), next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            torch.nn.utils.clip_grad_norm_(qf1.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(qf2.parameters(), 0.5)
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", current_q1.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", current_q2.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                print(
                    f"Step {global_step}: "
                    f"Q1 Loss = {qf1_loss.item():.3f}, "
                    f"Q2 Loss = {qf2_loss.item():.3f}, "
                    f"Actor Loss = {actor_loss.item():.3f}"
                )
                print("Q-values:", qf1_pi.mean().item(), qf2_pi.mean().item())
                print("Entropy:", -log_pi.mean().item())
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    envs.close()
    writer.close()