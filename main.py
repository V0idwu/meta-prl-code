import argparse
import copy
import os
import time
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.algos.ppo.ppo import PPOAgent
from utils.rl_utils import (
    Normalization,
    ReplayBuffer,
    RewardScaling,
    evaluate_policy,
    setup_config,
    try_gpu,
)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=227)
parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True)
parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--graph_depth", type=int, default=6)
parser.add_argument("--search_map_type", type=str, default="architecture")
parser.add_argument("--beta", type=float, default=0.5)
parser.add_argument("--arch_iter", type=float, default=1)
parser.add_argument("--prog_iter", type=float, default=1)
parser.add_argument("--max_train_steps", type=int, default=int(5e6))
parser.add_argument("--meta_iterations", type=int, default=250)
parser.add_argument("--inner_iterations", type=int, default=1)
parser.add_argument("--meta_lr", type=float, default=2e-2)
parser.add_argument("--max_episode_steps", type=int, default=500)
parser.add_argument("--batch_size", type=int, default=2048)
parser.add_argument("--mini_batch_size", type=int, default=64)
parser.add_argument("--hidden_dim", type=int, default=4)
parser.add_argument("--lr", type=float, default=4e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lamda", type=float, default=0.95)
parser.add_argument("--epsilon", type=float, default=0.2)
parser.add_argument("--K_epochs", type=int, default=5)
parser.add_argument("--use_adv_norm", type=lambda x: bool(strtobool(x)), default=True)
parser.add_argument("--use_state_norm", type=lambda x: bool(strtobool(x)), default=True)
parser.add_argument("--use_reward_scaling", type=lambda x: bool(strtobool(x)), default=True)
parser.add_argument("--entropy_coef", type=float, default=0.01)
parser.add_argument("--use_lr_decay", type=lambda x: bool(strtobool(x)), default=True)
parser.add_argument("--use_grad_clip", type=lambda x: bool(strtobool(x)), default=True)
parser.add_argument("--max_grad_norm", type=float, default=0.5)
parser.add_argument("--use_clip_v", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--target_kl", type=float, default=None)
args = parser.parse_args()
args.flip_fast_train_steps = args.max_train_steps // (args.meta_iterations * args.inner_iterations * 2)
args.unflip_fast_train_steps = args.max_train_steps // (args.meta_iterations * args.inner_iterations * 2)

device = try_gpu()
setup_config(seed=args.seed)


def orthogonal_init(layer, gain=np.sqrt(2)):
    for name, param in layer.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0)
        elif "weight" in name:
            nn.init.orthogonal_(param, gain=gain)

    return layer


class ActionCell(nn.Module):
    def __init__(self, index, args):
        super().__init__()

        self.index = index
        self.input_size = args.state_dim
        self.hidden_dim = args.hidden_dim
        self.output_size = args.action_dim

        self.W = nn.Linear(self.input_size, self.output_size)
        torch.nn.init.ones_(self.W.weight)

        self._hidden_state = None
        self.rnn = nn.GRU(self.input_size, self.output_size, batch_first=True)
        self.hidden_state_backup = None
        orthogonal_init(self.rnn, gain=0.1)

    @property
    def hidden_state(self):
        if self._hidden_state is not None:
            return self._hidden_state.detach()
        return self._hidden_state

    @hidden_state.setter
    def hidden_state(self, value):
        self._hidden_state = value

    def forward(self, x):
        x = x.view(-1, np.prod(self.input_size))
        # logit_tensor, self._hidden_state = self.rnn(x, self._hidden_state)
        # return logit_tensor
        return self.W(x)

    def reset_hidden_state(self):
        self._hidden_state = None

    def store_hidden_state(self):
        self.hidden_state_backup = self._hidden_state.detach()

    def restore_hidden_state(self):
        self._hidden_state = self.hidden_state_backup


class SymbolicCell(nn.Module):
    def __init__(self, index, args):
        super().__init__()
        self.index = index

        self.input_size = args.state_dim
        self.output_size = 1

        # self.P = nn.Linear(self.input_size, self.output_size, bias=True)

        self.rnn = nn.GRU(self.input_size, self.output_size, batch_first=True)
        self._hidden_state = None
        self.hidden_state_backup = None
        orthogonal_init(self.rnn, gain=0.1)

    @property
    def hidden_state(self):
        if self._hidden_state is not None:
            return self._hidden_state.detach()
        return self._hidden_state

    @hidden_state.setter
    def hidden_state(self, value):
        self._hidden_state = value

    def forward(self, x):
        x = x.view(-1, np.prod(self.input_size))
        x, self._hidden_state = self.rnn(x, self._hidden_state)
        # x = self.P(x)
        p_tensor = torch.sigmoid(x)
        return p_tensor

    def reset_hidden_state(self):
        self._hidden_state = None

    def store_hidden_state(self):
        self.hidden_state_backup = self._hidden_state.detach()

    def restore_hidden_state(self):
        self._hidden_state = self.hidden_state_backup


# nested ITE program
class Program(nn.Module):
    def __init__(self, obs_dim, act_dim, args, depth, device):
        super().__init__()
        self.device = device
        # self.envs = envs
        self.depth = depth

        self.action_space_dim = act_dim
        self.observation_space_dim = obs_dim
        self.args = args
        self.beta = args.beta

        self.cells = nn.ModuleList()
        self.num_cells = self.depth * 2 - 1
        self.hidden_state_backup = []

        for i in range(self.num_cells):
            if i % 2 == 0 and i != self.num_cells - 1:
                # 处理condition
                self.cells.append(SymbolicCell(i, args))
            else:
                # 处理controller
                self.cells.append(ActionCell(i, args))

        # P_map
        self.P_map = np.eye(self.depth, self.depth - 1)
        for i in range(self.P_map.shape[0]):
            for j in range(self.P_map.shape[1]):
                if i > j:
                    self.P_map[i, j] = -1

    def reset_hidden_state(self):
        for cell in self.cells:
            cell.reset_hidden_state()

    def store_hidden_state(self):
        for cell in self.cells:
            self.hidden_state_backup.append(cell.hidden_state)

    def restore_hidden_state(self):
        for i in range(self.num_cells):
            self.cells[i].hidden_state = self.hidden_state_backup[i]

    @staticmethod
    def get_action(model, x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action = model.act(x, deterministic=True)
        return torch.from_numpy(action)

    def forward(self, x):
        x = x.view(-1, self.args.state_dim)
        if self.depth == 1:
            W = [self.cells[0](x)]
            W_coefficient = [1]

        else:
            # compute without influence
            P = []  # symbolic cell
            W = []  # controller cell

            for i in range(self.num_cells):
                c = self.cells[i]
                if i % 2 == 0 and i != self.num_cells - 1:
                    P.append(c(x))
                else:
                    W.append(c(x))

            # compute with influence
            W_coefficient = [1 for i in range(len(W))]

            P_map = self.P_map
            for i in range(P_map.shape[0]):
                for j in range(P_map.shape[1]):
                    if P_map[i, j] == 1:
                        W_coefficient[i] *= P[j]
                    elif P_map[i, j] == -1:
                        W_coefficient[i] *= 1 - P[j]
                    elif P_map[i, j] == 0:
                        pass

        w = torch.zeros(x.shape[0], self.action_space_dim).to(self.device)

        for i in range(len(W)):
            w += W[i] * W_coefficient[i]

        prob_tensor = (w / self.beta).softmax(dim=1)

        # for i in range(self.num_models):
        #     action += w[:, i : i + 1] * self.get_action(self.models[i], x[:, self.index_action_space])

        return prob_tensor


class SimpleSearchMap(nn.Module):
    def __init__(self, depth):
        super().__init__()

        self.depth = depth
        self.type = "simple"

        # NOTE: for simplicity, we use softmax over a vector
        #       to represent the distribution of the programs.
        #       If the production rules of the DSL to expand
        #       programs are more than two, maintaining independent
        #       parameters for choosing programs in each layer
        #       of derivation graph is recommended.

        self.v = nn.Parameter(torch.zeros(self.depth), requires_grad=True)
        torch.nn.init.ones_(self.v)

    def freeze(self):
        self.v.requires_grad = False

    def unfreeze(self):
        self.v.requires_grad = True


class ArchitectureSearchMap(nn.Module):
    def __init__(self, depth):
        super().__init__()

        self.depth = depth
        self.type = "architecture"

        # Weight for each if-else probability
        self.options = nn.ParameterList()
        for _ in range(self.depth - 1):
            self.options.append(nn.Parameter(torch.rand(2), requires_grad=True))

    def freeze(self):
        for option in self.options:
            option.requires_grad = False

    def unfreeze(self):
        for option in self.options:
            option.requires_grad = True


# fusion ITE programs
# parameters
class FusionPrograms(nn.Module):
    def __init__(self, obs_dim, act_dim, args, depth, device):
        super().__init__()

        # self.envs = envs
        self.depth = depth
        self.action_space_dim = act_dim
        self.observation_space_dim = obs_dim

        self.beta = args.beta
        self.args = args
        self.device = device

        # shared cells
        self.shared_cells = nn.ModuleList()
        self.num_shared_cells = 2 * self.depth - 2
        self.hidden_state_backup = []

        for i in range(self.num_shared_cells):
            if i % 2 == 0:
                self.shared_cells.append(SymbolicCell(i, args))
            else:
                self.shared_cells.append(ActionCell(i, args))

        # exclusive cells
        self.ex_cells = nn.ModuleList()
        self.num_exclusive_cells = self.depth

        for i in range(self.num_exclusive_cells):
            self.ex_cells.append(ActionCell(i, args))

        # P_maps
        self.P_maps = []
        for d in range(self.depth):
            depth = d + 1
            P_map = np.eye(depth, depth - 1)
            for i in range(P_map.shape[0]):
                for j in range(P_map.shape[1]):
                    if i > j:
                        P_map[i, j] = -1
            self.P_maps.append(P_map)

    # @staticmethod
    # def get_action(model, x):
    #     with torch.no_grad():
    #         x = torch.as_tensor(x, dtype=torch.float32)
    #         action = model.act(x, deterministic=True)
    #     return torch.from_numpy(action)

    # @staticmethod
    # def get_action_and_value(self, x, action=None):
    #     logits = self.actor(x)
    #     probs = Categorical(logits=logits)
    #     if action is None:
    #         action = probs.sample()
    #     return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def reset_hidden_state(self):
        for cell in self.shared_cells:
            cell.reset_hidden_state()

        for cell in self.ex_cells:
            cell.reset_hidden_state()

    def store_hidden_state(self):
        for cell in self.shared_cells:
            self.hidden_state_backup.append(cell.hidden_state)

        for cell in self.ex_cells:
            self.hidden_state_backup.append(cell.hidden_state)

    def restore_hidden_state(self):
        for i in range(self.num_shared_cells):
            self.shared_cells[i].hidden_state = self.hidden_state_backup[i]

        for i in range(self.num_exclusive_cells):
            self.ex_cells[i].hidden_state = self.hidden_state_backup[i + self.num_shared_cells]

    def forward(self, x, search_map):
        x = x.view(-1, self.args.state_dim)
        # For SimpleSearchMap
        if search_map.type == "simple":
            v = F.softmax(search_map.v, dim=0)
        # For ArchitectureSearchMap
        elif search_map.type == "architecture":
            v = nn.Parameter(torch.ones(self.depth), requires_grad=False).to(self.device)
            for i in range(len(v)):
                options = search_map.options
                if i == 0:
                    v[i] = options[0].softmax(dim=0)[0]
                else:
                    prev = 1
                    for j in range(i):
                        prev *= options[j].softmax(dim=0)[1]
                    if i == len(v) - 1:
                        v[i] = prev
                    else:
                        option_value = options[i].softmax(dim=0)
                        v[i] = prev * option_value[0]

        prob_tensor = torch.zeros(x.shape[0], self.action_space_dim).to(self.device)

        if self.depth == 1:
            w = self.ex_cells[0](x)
            prob_tensor = (w / self.beta).softmax(dim=1)

        else:
            P = []  # symbolic cells
            A = []  # action cells
            ex_A = []

            for i in range(self.num_shared_cells):
                c = self.shared_cells[i]
                if i % 2 == 0:
                    P.append(c(x))
                else:
                    A.append(c(x))

            for i in range(self.num_exclusive_cells):
                c = self.ex_cells[i]
                ex_A.append(c(x))

            for d in range(self.depth):
                depth = d + 1
                if depth == 1:
                    w = ex_A[0]
                    w = (w / self.beta).softmax(dim=1)
                    prob_tensor += v[0] * w

                else:
                    # independent P_map and W_coefficient for each program
                    P_map = self.P_maps[d]

                    W_coefficient = [1 for i in range(P_map.shape[0])]
                    for i in range(P_map.shape[0]):
                        for j in range(P_map.shape[1]):
                            if P_map[i, j] == 1:
                                W_coefficient[i] *= P[j]
                            elif P_map[i, j] == -1:
                                W_coefficient[i] *= 1 - P[j]
                            elif P_map[i, j] == 0:
                                pass

                    w = torch.zeros(x.shape[0], self.action_space_dim).to(self.device)

                    for i in range(len(W_coefficient) - 1):
                        w += A[i] * W_coefficient[i]
                    w += ex_A[d] * W_coefficient[i + 1]
                    w = (w / self.beta).softmax(dim=1)
                    prob_tensor += v[d] * w

        return prob_tensor

    def freeze(self):
        for cell in self.shared_cells:
            for param in cell.parameters():
                param.requires_grad = False

        for cell in self.ex_cells:
            for param in cell.parameters():
                param.requires_grad = False

    def unfreeze(self):
        for cell in self.shared_cells:
            for param in cell.parameters():
                param.requires_grad = True

        for cell in self.ex_cells:
            for param in cell.parameters():
                param.requires_grad = True


# program derivation graph
class SearchFusionProgram(nn.Module):
    def __init__(self, obs_dim, act_dim, args, device):
        super().__init__()

        # NOTE: graph_depth is different from program AST depth,
        #       e.g., when graph_depth is set to 2, our algorithm
        #       would only search for a single program with AST
        #       depth = 1, i.e., a simple low-level controller.
        #       For  the program AST depth, "if" is on upper
        #       level, "then" and "else" are on the lower level.

        # 默认graph_depth至少为2，即AST_depth至少为1
        assert args.graph_depth >= 2
        self.graph_depth = args.graph_depth
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        # self.envs = envs
        self.args = args
        self.device = device
        self.search_map_type = args.search_map_type  # weights for architecture
        if self.search_map_type == "architecture":
            self.search_map = ArchitectureSearchMap(depth=self.graph_depth - 1)
        elif self.search_map_type == "simple":
            self.search_map = SimpleSearchMap(depth=self.graph_depth - 1)
        self.fusion_programs = FusionPrograms(self.obs_dim, self.act_dim, args, depth=self.graph_depth - 1, device=device)
        self.search_map.unfreeze()
        self.fusion_programs.freeze()
        self.pointer = 0  # 0 for optimizing architecture, 1 for optimizing programs

    def flip(self):
        if self.pointer == 0:
            self.pointer = 1
            self.search_map.freeze()
            self.fusion_programs.unfreeze()
        elif self.pointer == 1:
            self.pointer = 0
            self.search_map.unfreeze()
            self.fusion_programs.freeze()

    def forward(self, x):
        action_logits = self.fusion_programs(x, self.search_map)
        return action_logits

    def reset_hidden_state(self):
        self.fusion_programs.reset_hidden_state()

    def store_hidden_state(self):
        self.fusion_programs.store_hidden_state()

    def restore_hidden_state(self):
        self.fusion_programs.restore_hidden_state()

    # return a discrete program
    def extract(self):
        """
        search for maximal architecture
        """
        if self.search_map_type == "simple":
            index = self.search_map.v.argmax()
        elif self.search_map_type == "architecture":
            v = nn.Parameter(torch.ones(self.graph_depth - 1), requires_grad=False)
            for i in range(len(v)):
                options = self.search_map.options
                if i == 0:
                    v[i] = options[0].softmax(dim=0)[0]
                else:
                    prev = 1
                    for j in range(i):
                        prev *= options[j].softmax(dim=0)[1]
                    if i == len(v) - 1:
                        v[i] = prev
                    else:
                        option_value = options[i].softmax(dim=0)
                        v[i] = prev * option_value[0]
            index = v.argmax()

        prog = Program(self.obs_dim, self.act_dim, self.args, depth=index + 1, device=self.device)

        # empty the cells
        prog.cells = nn.ModuleList()

        self.fusion_programs.unfreeze()
        # extract shared cells
        for i in range(2 * index):
            prog.cells.append(copy.deepcopy(self.fusion_programs.shared_cells[i]))

        # extract exclusive cell
        prog.cells.append(copy.deepcopy(self.fusion_programs.ex_cells[index]))

        return prog, index


# class ProgAgent(nn.Module):
#     def __init__(self, obs_dim, prog):
#         super().__init__()
#         self.critic = CriticNet(input_size=obs_dim)
#         # self.actor = env_nets.ActorNet()
#         self.actor = prog

#     def get_value(self, x):
#         return self.critic(x)

#     def get_action_and_value(self, x, action=None):
#         logits = self.actor(x)
#         probs = Categorical(logits=logits)
#         if action is None:
#             action = probs.sample()
#         return action, probs.log_prob(action), probs.entropy(), self.critic(x)

#     def update_prog_network(self, prog):

#         self.model = prog
#         # self.log_std = Variable(torch.ones(self.m) * init_log_std, requires_grad=True)
#         self.trainable_params = [p for p in list(self.model.parameters()) if p.requires_grad == True] + [self.log_std]

#         # Old Policy network
#         # ------------------------
#         self.old_model = copy.deepcopy(prog)
#         # self.old_log_std = Variable(torch.ones(self.m) * init_log_std)
#         self.old_params = [p for p in list(self.old_model.parameters()) if p.requires_grad == True] + [self.old_log_std]

#         for idx, param in enumerate(self.old_params):
#             param.data = self.trainable_params[idx].data.clone()

#         # Easy access variables
#         # -------------------------
#         # self.log_std_val = np.float64(self.log_std.data.numpy().ravel())
#         self.param_shapes = [p.data.numpy().shape for p in self.trainable_params]
#         self.param_sizes = [p.data.numpy().size for p in self.trainable_params]
#         self.d = np.sum(self.param_sizes)  # total number of params


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)
        self.activate_func = nn.Tanh()

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s


def do_fast_weight_update(agent, env, args, device, eval=False, **kwargs):
    step_info = kwargs.get("step_info")
    inner_iterations = kwargs.get("inner_iterations")
    prog = SearchFusionProgram(obs_dim=env.observation_space.shape, act_dim=env.action_space.n, args=args, device=device)
    fast_agent = PPOAgent(args, prog, Critic(args), device)
    fast_agent.ac.load_state_dict(agent.ac.state_dict())
    # fast_agent.critic.load_state_dict(agent.critic.state_dict())

    replay_buffer = ReplayBuffer(args)
    state_norm = Normalization(shape=args.state_dim)
    reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    for _ in range(inner_iterations):
        step_info["fast_steps"] = 0

        while step_info["fast_steps"] < args.max_fast_train_steps:
            s = env.reset()
            if args.use_state_norm:
                s = state_norm(s)
            if args.use_reward_scaling:
                reward_scaling.reset()
            episode_steps = 0
            done = False
            while not done and episode_steps <= args.episode_steps_limit:
                episode_steps += 1
                a, a_logprob = fast_agent.choose_action(s)
                s_, r, done, _ = env.step(a)

                if args.use_state_norm:
                    s_ = state_norm(s_)
                if args.use_reward_scaling:
                    r = reward_scaling(r)

                if done and episode_steps != args.episode_steps_limit:
                    dw = True
                else:
                    dw = False

                replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
                s = s_
                step_info["fast_steps"] += 1
                if not eval:
                    step_info["global_steps"] += 1

                if replay_buffer.count == args.batch_size:
                    for _ in range(args.arch_iter):
                        agent.train(replay_buffer, step_info)
                        agent.ac.actor.flip()

                    for _ in range(args.prog_iter):
                        agent.train(replay_buffer, step_info)
                        agent.ac.actor.flip()
                    replay_buffer.count = 0

    return fast_agent, state_norm


def train_by_curriculum_learning(level_range, cur, total):
    percentile_threshold = 0.7
    percentile = cur / total
    if percentile > percentile_threshold:
        level = np.random.choice(level_range, size=1).item()
    else:
        stride = percentile_threshold / len(level_range)
        level = level_range[-1]
        for i in level_range:
            if percentile < i * stride:
                level = i
                break
    return level


def main(env_type, args, device):
    env = env_type()
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.n

    replay_buffer = ReplayBuffer(args)
    prog = SearchFusionProgram(obs_dim=env.observation_space.shape, act_dim=env.action_space.n, args=args, device=device)
    agent = PPOAgent(args, actor=prog, critic=Critic(args), device=device)
    meta_optimizer = torch.optim.Adam(agent.ac.parameters(), lr=args.meta_lr, eps=1e-5)

    state_norm = Normalization(shape=args.state_dim)
    reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    evaluate_num = 0
    step_info = {
        "global_steps": 0,
        "fast_steps": 0,
        "max_train_steps": args.max_train_steps,
        "flip_train_steps": args.flip_train_steps,
        "unflip_train_steps": args.unflip_train_steps,
    }

    for meta_steps in range(1, int(args.meta_iterations // 2) + 1):
        # update learning rate of meta optimizer
        lr_now = args.meta_lr * (1 - meta_steps / args.meta_iterations)
        for p in meta_optimizer.param_groups:
            p["lr"] = lr_now

        if True:
            level = train_by_curriculum_learning(args.level_range, meta_steps, args.meta_iterations // 2)
        else:
            level = np.random.choice(args.level_range, size=1).item()

        env = args.env_type(level)

        # sample envs && do fast weight learning
        fast_agent, state_norm = do_fast_weight_update(
            agent, env, args, device, step_info=step_info, inner_iterations=args.inner_iterations
        )

        # update slow weight
        # Inject updates into each .grad
        for p, fast_p in zip(agent.ac.parameters(), fast_agent.ac.parameters()):
            if p.grad is None:
                p.grad = torch.zeros(p.size(), requires_grad=True).to(device)
            p.grad.data.add_(p.data - fast_p.data)

        # Update meta-parameters
        meta_optimizer.step()
        meta_optimizer.zero_grad()

        # Evaluate the policy
        if args.eval_model:
            finetune_agent, state_norm = do_fast_weight_update(
                agent, env, args, device, eval=True, step_info=step_info, inner_iterations=4
            )
            info = evaluate_policy(args, finetune_agent, state_norm=state_norm)

        agent.reset_hidden_state()

    extracted_prog, index = agent.ac.actor.extract()
    print(f"index={index}")
    agent = PPOAgent(args, actor=extracted_prog, critic=Critic(args), device=device)
    meta_optimizer = torch.optim.Adam(agent.ac.actor.parameters(), lr=args.meta_lr, eps=1e-5)

    for meta_steps in range(int(args.meta_iterations // 2) + 1, int(args.meta_iterations) + 1):
        lr_now = args.meta_lr * (1 - meta_steps / args.meta_iterations)
        for p in meta_optimizer.param_groups:
            p["lr"] = lr_now

        level = np.random.choice(args.level_range, size=1).item()
        env = args.env_type(level)

        fast_agent, state_norm = do_fast_weight_update(
            agent,
            env,
            args,
            device,
            flip=False,
            step_info=step_info,
            max_fast_train_steps=args.unflip_fast_train_steps,
            index=index,
        )

        for p, fast_p in zip(agent.ac.parameters(), fast_agent.ac.parameters()):
            if p.grad is None:
                p.grad = torch.zeros(p.size(), requires_grad=True).to(device)
            p.grad.data.add_(p.data - fast_p.data)

        meta_optimizer.step()
        meta_optimizer.zero_grad()

        if args.eval_model:
            agent.store_hidden_state()
            finetune_agent, state_norm = do_fast_weight_update(
                agent,
                env,
                args,
                device,
                eval=True,
                flip=False,
                step_info=step_info,
                inner_iterations=4,
                max_fast_train_steps=args.unflip_fast_train_steps,
                index=index,
            )
            info = evaluate_policy(args, finetune_agent, state_norm=state_norm)
            agent.restore_hidden_state()

            print(
                f"global steps: {step_info['global_steps']:,} | unflip meta_steps: {meta_steps} | reward: {info[''.join([str(i) for i in args.eval_level_range])][0]:.2f} | steps: {info[''.join([str(i) for i in args.eval_level_range])][1]:.2f}"
            )

        agent.reset_hidden_state()
