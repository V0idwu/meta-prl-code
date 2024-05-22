import collections
import gc
import os
import random
import socket

import gymnasium as gym
import imageio
import numpy as np
import torch
from stable_baselines3.common.env_checker import check_env


def setup_config(seed: int = 227):
    """set random seed for pytorch, numpy and random library."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.set_default_dtype(torch.float32)


def try_gpu(i: int = 0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")


# from .base_utils import make_dir
class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)

    def load(self, **kwargs):
        raise NotImplementedError()


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)


def make_env(env_type, env_scale, seed):
    def thunk():
        env = env_type(env_scale)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def check_gym_env(env: gym.Env) -> None:
    print(f"env.observation_space.shape: {env.observation_space.shape}, sample: {env.observation_space.sample()}")
    print(f"env.action_space.n: {env.action_space.n}, sample: {env.action_space.sample()}")
    print(f"stable_baselines3.check_env: {check_env(env)}")


# def save_image(path: str, img: np.ndarray) -> None:
#     make_dir(os.path.split(path)[0])
#     imageio.imsave(path, img)


# def _sinle_env_evaluate_policy(args, agent, env, state_norm):
#     s = env.reset()
#     if args.use_state_norm:  # During the evaluating,update=False
#         s = state_norm(s, update=False)
#     done = False
#     reward = 0
#     steps = 0
#     while not done and steps <= args.episode_steps_limit:
#         a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
#         s_, r, done, _ = env.step(a)
#         if args.use_state_norm:
#             s_ = state_norm(s_, update=False)
#         reward += r
#         steps += 1
#         s = s_

#     return reward, steps


def _policy_rollout_eval_continuous(agent, env, seed, use_state_norm, state_norm, use_rnn, policy_dist, max_action):
    state, _ = env.reset(seed)
    if type(state) == dict:
        state = np.concatenate([state["observation"], state["achieved_goal"]])
    if use_state_norm:  # During the evaluating, update=False
        state = state_norm(state, update=False)
    done = False
    episode_reward = 0
    episode_steps = 0
    if use_rnn:
        agent.reset_hidden_state()
    while not done:
        action = agent.evaluate(state)  # We use the deterministic policy during the evaluating
        if policy_dist == "Beta":
            action = 2 * (action - 0.5) * max_action  # [0,1]->[-max,max]
        next_state, reward, terminated, truncated, _ = env.step(action)
        if type(next_state) == dict:
            next_state = np.concatenate([next_state["observation"], next_state["achieved_goal"]])
        done = terminated or truncated
        if use_state_norm:
            next_state = state_norm(next_state, update=False)
        episode_reward += reward
        episode_steps += 1
        state = next_state
    return episode_reward, episode_steps


def _policy_rollout_eval_discrete(agent, env, seed, use_state_norm, state_norm, use_rnn):
    state, _ = env.reset(seed)
    if use_state_norm:  # During the evaluating, update=False
        state = state_norm(state, update=False)
    done = False
    episode_reward = 0
    episode_steps = 0
    if use_rnn:
        agent.reset_hidden_state()
    while not done:
        action = agent.evaluate(state)  # We use the deterministic policy during the evaluating
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if use_state_norm:
            next_state = state_norm(next_state, update=False)
        episode_reward += reward
        episode_steps += 1
        state = next_state

    return episode_reward, episode_steps


def evaluate_policy(args, agent, use_rnn=False, **kwargs):
    state_norm = kwargs.get("state_norm", None)
    eval_times = kwargs.get("eval_times", 10)
    use_rnn = kwargs.get("use_rnn", False)

    eval_ret = {}
    agent.ac.eval()
    for i in args.eval_level_range:
        reward_lst = []
        steps_lst = []
        env = args.env_type(num_goals=i, max_episode_steps=args.max_episode_steps)
        for j in range(eval_times):
            if type(env.action_space) == gym.spaces.discrete.Discrete:
                episode_reward, episode_steps = _policy_rollout_eval_discrete(
                    agent, env, (args.seed + i + j), args.use_state_norm, state_norm, use_rnn
                )
            else:
                episode_reward, episode_steps = _policy_rollout_eval_continuous(
                    agent, env, (args.seed + i + j), args.use_state_norm, state_norm, use_rnn, args.policy_dist, args.max_action
                )
            reward_lst.append(episode_reward)
            steps_lst.append(episode_steps)

        eval_ret[str(i)] = (np.mean(reward_lst), np.mean(steps_lst))
        env.close()
        del env
        gc.collect()

    return eval_ret


def evaluate_policy_rnn(args, agent, **kwargs):
    state_norm = kwargs.get("state_norm")
    eval_times = kwargs.get("eval_times") if kwargs.get("eval_times") is not None else 5

    info = {}

    for i in args.eval_level_range:
        eval_rewards = []
        eval_steps = []
        env = args.env_type(i)
        seed = args.seed + 100
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        for _ in range(eval_times):
            reward, steps, done = 0, 0, False
            s = env.reset()
            agent.reset_rnn_hidden()  # reset rnn hidden
            while not done and steps <= args.episode_steps_limit:
                if args.use_state_norm:
                    s = state_norm(s, update=False)
                a, a_logprob = agent.choose_action(s, evaluate=True)
                s_, r, done, _ = env.step(a)
                reward += r
                steps += 1
                s = s_

            eval_rewards.append(reward)
            eval_steps.append(steps)

            info[str(i)] = (np.mean(eval_rewards), np.mean(eval_steps))

    total_reward = []
    total_steps = []
    for k in info:
        total_reward.append(info[k][0])
        total_steps.append(info[k][1])

    info["".join(info.keys())] = (np.mean(total_reward), np.mean(total_steps))

    return info


# class EvaluatePolicy:
#     EVAL_NUM_EPISODES = 100
#     EVAL_NUM_STEPS = 200
#     EVAL_NUM_ENVS = 16

#     def __init__(self, agent: torch.nn.Module, args: argparse.Namespace, **kwargs):
#         self.agent = agent
#         self.args = args
#         self.device = kwargs.get("device") if kwargs.get("device") is not None else torch.device("cpu")
#         self.eval_num_episodes = (
#             kwargs.get("num_episodes") if kwargs.get("num_episodes") is not None else self.EVAL_NUM_EPISODES
#         )
#         self.eval_num_steps = kwargs.get("num_steps") if kwargs.get("num_steps") is not None else self.EVAL_NUM_STEPS
#         self.eval_num_envs = kwargs.get("num_envs") if kwargs.get("num_envs") is not None else self.EVAL_NUM_ENVS

#         self.evaluate_info = {
#             "exp_name": [],
#             "env_id": [],
#             "level": [],
#             "num_steps": [],
#             "total_num_episodes": [],
#             "avg_episode_step": [],
#             "avg_episode_reward": [],
#         }

#     def eval(self, state_norm=None, times=1):
#         # env = self.args.env_type(self.args.level)  # env for eval
#         envs = gym.vector.SyncVectorEnv(
#             [make_env(self.args.env_type, self.args.level, 0 + i) for i in range(self.args.num_envs)]
#         )
#         episode_reward_lst = []
#         for _ in range(times):
#             rewards = np.zeros((self.args.num_steps, self.args.num_envs))
#             dones = np.zeros((self.args.num_steps, self.args.num_envs))
#             s = envs.reset()
#             if self.args.use_state_norm:  # During the evaluating,update=False
#                 s = state_norm(s, update=False)
#             for i in range(self.args.num_steps):
#                 a = self.agent.evaluate(s, self.device)  # We use the deterministic policy during the evaluating
#                 s_, r, done, _ = envs.step(a)
#                 if self.args.use_state_norm:
#                     s_ = state_norm(s_, update=False)
#                 s = s_
#                 rewards[i] = r
#                 dones[i] = done

#             valid_step_index = np.ones((self.args.num_envs,)) * (self.args.num_steps - 1)
#             row, col = np.nonzero(dones)
#             for i, j in zip(row, col):
#                 if i < valid_step_index[j]:
#                     valid_step_index[j] = i
#             # episode_step = [int(row) + 1 for row in valid_step_index]
#             episode_reward = [rewards[0 : int(step_index) + 1, col].sum() for col, step_index in enumerate(valid_step_index)]
#             episode_reward_lst.extend(episode_reward)

#         return np.mean(episode_reward_lst)

#     def evalute(self, verbose=True):
#         for level in self.level_range:
#             self.evaluate_info["env_id"].append(self.env_type.id)
#             self.evaluate_info["level"].append(level)
#             self.evaluate_info["num_steps"].append(self.eval_num_steps)
#             self.evaluate_info["total_num_episodes"].append(self.eval_num_episodes * self.NUM_ENVS)
#             self.evaluate_info["exp_name"].append(self.exp_name)

#             self.envs = gym.vector.SyncVectorEnv([make_env(self.env_type, level, 0 + i) for i in range(self.NUM_ENVS)])

#             episode_step_lst, episode_reward_lst = self._policy_rollout_eval()

#             # evaluate_info["evaluate_result"] = {"episode_step": episode_step_lst, "episode_reward": episode_reward_lst}
#             self.evaluate_info["avg_episode_step"].append(np.mean(episode_step_lst))
#             self.evaluate_info["avg_episode_reward"].append(np.mean(episode_reward_lst))

#         if verbose:
#             print(tabulate(self.evaluate_info, tablefmt="plain", headers="keys", showindex="always"))

#     def _policy_rollout_eval(self) -> Tuple[list, list]:

#         episode_reward_lst = []
#         episode_step_lst = []
#         for _ in range(self.eval_num_episodes):
#             rewards = np.zeros((self.eval_num_steps, self.eval_num_envs))
#             dones = np.zeros((self.eval_num_steps, self.eval_num_envs))
#             obs = self.envs.reset()
#             for i in range(self.eval_num_steps):
#                 if self.agent is None:
#                     action = self.envs.action_space.sample()
#                 else:
#                     action, *_ = self.agent.get_action_and_value(torch.Tensor(obs).to(self.device))
#                     action = action.cpu().numpy()
#                 obs, reward, done, info = self.envs.step(action)
#                 rewards[i] = reward
#                 dones[i] = done

#             valid_step_index = np.ones((self.eval_num_envs,)) * (self.eval_num_steps - 1)
#             row, col = np.nonzero(dones)
#             for i, j in zip(row, col):
#                 if i < valid_step_index[j]:
#                     valid_step_index[j] = i
#             episode_step = [int(row) + 1 for row in valid_step_index]
#             episode_reward = [rewards[0 : int(step_index) + 1, col].sum() for col, step_index in enumerate(valid_step_index)]

#             episode_reward_lst.extend(episode_reward)
#             episode_step_lst.extend(episode_step)

#         return episode_step_lst, episode_reward_lst


class EarlyStopping:
    def __init__(self, patience=30, delta=1, **kwargs):
        # self.save_path = save_path
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.logger = kwargs.get("logger", None)

    def __call__(self, steps):
        score = steps
        if self.best_score is None:
            self.best_score = score

        elif score >= self.best_score - self.delta and score <= self.best_score + self.delta:
            self.counter += 1
            if self.logger is not None:
                self.logger.debug(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        elif score < self.best_score - self.delta:
            self.best_score = score
            self.counter = 0

        else:
            self.counter = 0


class ReplayBuffer:
    def __init__(self, batch_size, state_dim, action_dim=1):
        self.s = np.zeros((batch_size, state_dim))
        self.a = np.zeros((batch_size, action_dim))
        self.a_logprob = np.zeros((batch_size, action_dim))
        self.r = np.zeros((batch_size, 1))
        self.s_ = np.zeros((batch_size, state_dim))
        self.dw = np.zeros((batch_size, 1))
        self.done = np.zeros((batch_size, 1))
        self.count = 0

    def store(self, s, a, a_logprob, r, s_, dw, done):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def size(self):
        return self.count

    def return_all_samples(self, device):
        s = torch.tensor(self.s, dtype=torch.float).to(device)
        a = torch.tensor(self.a, dtype=torch.long).to(device)  # In discrete action space, 'a' needs to be torch.long
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float).to(device)
        r = torch.tensor(self.r, dtype=torch.float).to(device)
        s_ = torch.tensor(self.s_, dtype=torch.float).to(device)
        dw = torch.tensor(self.dw, dtype=torch.float).to(device)
        done = torch.tensor(self.done, dtype=torch.float).to(device)

        return s, a, a_logprob, r, s_, dw, done

    def clear(self):
        self.count = 0


# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.buffer = collections.deque(maxlen=capacity)

#     def store(self, state, action, action_logprob, reward, next_state, dw, done):
#         self.buffer.append((state, action, action_logprob, reward, next_state, dw, done))

#     def size(self):
#         return len(self.buffer)

#     def return_all_samples(self, device):
#         all_transitions = list(self.buffer)
#         state, action, action_logprob, reward, next_state, dw, done = zip(*all_transitions)
#         s = torch.from_numpy(np.array(state, dtype=np.float32)).to(device)
#         a = torch.from_numpy(np.array(action, dtype=np.int64)).to(device)
#         a_logprob = torch.from_numpy(np.array(action_logprob, dtype=np.float32)).to(device)
#         r = torch.from_numpy(np.array(reward, dtype=np.float32)).to(device)
#         s_ = torch.from_numpy(np.array(next_state, dtype=np.float32)).to(device)
#         dw = torch.from_numpy(np.array(dw, dtype=np.float32)).to(device)
#         done = torch.from_numpy(np.array(done, dtype=np.float32)).to(device)
#         return s, a, a_logprob, r, s_, dw, done

#     def reset(self):
#         self.buffer.clear()


class SumTree(object):
    """
    Story data with its priority in the tree.
    Tree structure and array storage:

    Tree index:
         0         -> storing priority sum
        / \
      1     2
     / \   / \
    3   4 5   6    -> storing priority for transitions

    Array type for storing:
    [0,1,2,3,4,5,6]
    """

    def __init__(self, buffer_capacity):
        self.buffer_capacity = buffer_capacity  # buffer的容量
        self.tree_capacity = 2 * buffer_capacity - 1  # sum_tree的容量
        self.tree = np.zeros(self.tree_capacity)

    def update(self, data_index, priority):
        # data_index表示当前数据在buffer中的index
        # tree_index表示当前数据在sum_tree中的index
        tree_index = data_index + self.buffer_capacity - 1  # 把当前数据在buffer中的index转换为在sum_tree中的index
        change = priority - self.tree[tree_index]  # 当前数据的priority的改变量
        self.tree[tree_index] = priority  # 更新树的最后一层叶子节点的优先级
        # then propagate the change through the tree
        while tree_index != 0:  # 更新上层节点的优先级，一直传播到最顶端
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_index(self, v):
        parent_idx = 0  # 从树的顶端开始
        while True:
            child_left_idx = 2 * parent_idx + 1  # 父节点下方的左右两个子节点的index
            child_right_idx = child_left_idx + 1
            if child_left_idx >= self.tree_capacity:  # reach bottom, end search
                tree_index = parent_idx  # tree_index表示采样到的数据在sum_tree中的index
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[child_left_idx]:
                    parent_idx = child_left_idx
                else:
                    v -= self.tree[child_left_idx]
                    parent_idx = child_right_idx

        data_index = tree_index - self.buffer_capacity + 1  # tree_index->data_index
        return data_index, self.tree[tree_index]  # 返回采样到的data在buffer中的index,以及相对应的priority

    def get_batch_index(self, current_size, batch_size, beta):
        batch_index = np.zeros(batch_size, dtype=np.int64)
        IS_weight = torch.zeros(batch_size, dtype=torch.float32)
        segment = self.priority_sum / batch_size  # 把[0,priority_sum]等分成batch_size个区间，在每个区间均匀采样一个数
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            index, priority = self.get_index(v)
            batch_index[i] = index
            prob = priority / self.priority_sum  # 当前数据被采样的概率
            IS_weight[i] = (current_size * prob) ** (-beta)
        IS_weight /= IS_weight.max()  # normalization

        return batch_index, IS_weight

    @property
    def priority_sum(self):
        return self.tree[0]  # 树的顶端保存了所有priority之和

    @property
    def priority_max(self):
        return self.tree[self.buffer_capacity - 1 :].max()  # 树的最后一层叶节点，保存的才是每个数据对应的priority


class N_Steps_Prioritized_ReplayBuffer(object):
    def __init__(self, args):
        self.max_train_steps = args.max_train_steps
        self.alpha = args.alpha
        self.beta_init = args.beta_init
        self.beta = args.beta_init
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.buffer_capacity = args.buffer_capacity
        self.sum_tree = SumTree(self.buffer_capacity)
        self.n_steps = args.n_steps
        self.n_steps_deque = collections.deque(maxlen=self.n_steps)
        self.buffer = {
            "state": np.zeros((self.buffer_capacity, args.state_dim)),
            "action": np.zeros((self.buffer_capacity, 1)),
            "reward": np.zeros(self.buffer_capacity),
            "next_state": np.zeros((self.buffer_capacity, args.state_dim)),
            "terminal": np.zeros(self.buffer_capacity),
        }
        self.current_size = 0
        self.count = 0

    def store_transition(self, state, action, reward, next_state, terminal, done):
        transition = (state, action, reward, next_state, terminal, done)
        self.n_steps_deque.append(transition)
        if len(self.n_steps_deque) == self.n_steps:
            state, action, n_steps_reward, next_state, terminal = self.get_n_steps_transition()
            self.buffer["state"][self.count] = state
            self.buffer["action"][self.count] = action
            self.buffer["reward"][self.count] = n_steps_reward
            self.buffer["next_state"][self.count] = next_state
            self.buffer["terminal"][self.count] = terminal
            # 如果是buffer中的第一条经验，那么指定priority为1.0；否则对于新存入的经验，指定为当前最大的priority
            priority = 1.0 if self.current_size == 0 else self.sum_tree.priority_max
            self.sum_tree.update(data_index=self.count, priority=priority)  # 更新当前经验在sum_tree中的优先级
            self.count = (self.count + 1) % self.buffer_capacity  # When 'count' reaches buffer_capacity, it will be reset to 0.
            self.current_size = min(self.current_size + 1, self.buffer_capacity)

    def sample(self, total_steps, device):
        batch_index, IS_weight = self.sum_tree.get_batch_index(
            current_size=self.current_size, batch_size=self.batch_size, beta=self.beta
        )
        self.beta = self.beta_init + (1 - self.beta_init) * (total_steps / self.max_train_steps)  # beta：beta_init->1.0
        batch = {}
        for key in self.buffer.keys():  # numpy->tensor
            if key == "action":
                batch[key] = torch.tensor(self.buffer[key][batch_index], dtype=torch.long).to(device)
            else:
                batch[key] = torch.tensor(self.buffer[key][batch_index], dtype=torch.float32).to(device)

        return batch, batch_index, IS_weight.to(device)

    def get_n_steps_transition(self):
        state, action = self.n_steps_deque[0][:2]  # 获取deque中第一个transition的s和a
        next_state, terminal = self.n_steps_deque[-1][3:5]  # 获取deque中最后一个transition的s'和terminal
        n_steps_reward = 0
        for i in reversed(range(self.n_steps)):  # 逆序计算n_steps_reward
            r, s_, ter, d = self.n_steps_deque[i][2:]
            n_steps_reward = r + self.gamma * (1 - d) * n_steps_reward
            if (
                d
            ):  # 如果done=True，说明一个回合结束，保存deque中当前这个transition的s'和terminal作为这个n_steps_transition的next_state和terminal
                next_state, terminal = s_, ter

        return state, action, n_steps_reward, next_state, terminal

    def update_batch_priorities(self, batch_index, td_errors):  # 根据传入的td_error，更新batch_index所对应数据的priorities
        priorities = (np.abs(td_errors) + 0.01) ** self.alpha
        for index, priority in zip(batch_index, priorities):
            self.sum_tree.update(data_index=index, priority=priority)
