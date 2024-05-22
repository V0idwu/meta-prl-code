import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

# from nets import res_linear


class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super(ActorCritic, self).__init__()
        self.actor = actor
        self.critic = critic

    def reset_hidden_state(self):
        self.actor.reset_hidden_state()
        self.critic.reset_hidden_state()

    def store_hidden_state(self):
        self.actor.store_hidden_state()
        self.critic.store_hidden_state()

    def restore_hidden_state(self):
        self.actor.restore_hidden_state()
        self.critic.restore_hidden_state()


class PPOAgent:
    def __init__(self, args, actor, critic, device, writer=None):
        self.args = args
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        # self.lr = args.lr
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.use_grad_clip = args.use_grad_clip
        self.max_grad_norm = args.max_grad_norm
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.use_clip_v = args.use_clip_v
        self.epsilon = args.epsilon

        self.ac = ActorCritic(actor, critic).to(device)
        # self.optimizer = torch.optim.Adam(
        #     [
        #         {"params": self.ac.actor.parameters()},
        #         {"params": self.ac.critic.parameters()},
        #     ],
        #     lr=self.lr,
        #     eps=1e-5,
        # )

        self.optimizer_actor = torch.optim.Adam(self.ac.actor.parameters(), lr=self.lr_a, eps=1e-5)
        self.optimizer_critic = torch.optim.Adam(self.ac.critic.parameters(), lr=self.lr_c, eps=1e-5)

        self.device = device

    # def evaluate(self, s):  # When evaluating the policy, we select the action with the highest probability
    #     s = torch.tensor(s, dtype=torch.float).unsqueeze(0).to(self.device)
    #     a_prob = self.ac.actor_forward(s).detach().cpu().numpy().flatten()
    #     a = np.argmax(a_prob)
    #     return a

    def reset_hidden_state(self):
        self.ac.reset_hidden_state()

    def store_hidden_state(self):
        self.ac.store_hidden_state()

    def restore_hidden_state(self):
        self.ac.restore_hidden_state()

    def evaluate(self, s):  # When evaluating the policy, we select the action with the highest probability
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)
        a_prob = self.ac.actor(s).detach().cpu().numpy().flatten()
        a = np.argmax(a_prob)
        return a

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)
        with torch.no_grad():
            dist = Categorical(probs=self.ac.actor(s))
            a = dist.sample()
            a_logprob = dist.log_prob(a)
        return a.cpu().numpy()[0], a_logprob.cpu().numpy()[0]

    # def choose_action(self, s, evaluate=False):
    #     with torch.no_grad():
    #         s = torch.tensor(s, dtype=torch.float).unsqueeze(0).to(self.device)
    #         a_prob = self.ac.actor(s)
    #         if evaluate:
    #             a = torch.argmax(a_prob)
    #             return a.detach().cpu().item(), None
    #         else:
    #             dist = Categorical(probs=a_prob)
    #             a = dist.sample()
    #             a_logprob = dist.log_prob(a)
    #             return a.detach().cpu().item(), a_logprob.detach().cpu().item()

    def update(self, replay_buffer, cur_steps, total_steps):
        s, a, a_logprob, r, s_, dw, done = replay_buffer.return_all_samples(self.device)  # Get training data
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the episode_steps_limit). When calculating the adv, if done=True, gae=0
        """
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.ac.critic(s)
            vs_ = self.ac.critic(s_)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().cpu().numpy()), reversed(done.flatten().cpu().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1).to(self.device)
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                # self.reset_hidden_state()

                dist_now = Categorical(probs=self.ac.actor(s[index]))
                dist_entropy = dist_now.entropy().view(-1, 1)  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index].squeeze()).view(-1, 1)  # shape(mini_batch_size X 1)
                # a/b=exp(log(a)-log(b))
                log_ratios = a_logprob_now - a_logprob[index]  # shape(mini_batch_size X 1)
                ratios = torch.exp(log_ratios)
                approx_kl = ((ratios - 1) - log_ratios).mean()

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # shape(mini_batch_size X 1)

                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.ac.actor.parameters(), self.max_grad_norm)
                self.optimizer_actor.step()

                # Update critic
                v_s = self.ac.critic(s[index])
                if self.use_clip_v:
                    # clip v
                    critic_loss_unclipped = (v_s - v_target[index]) ** 2
                    critic_clipped = v_target[index] + torch.clamp(v_s - v_target[index], -self.epsilon, self.epsilon)
                    critic_loss_clipped = (critic_clipped - v_target[index]) ** 2
                    critic_loss_max = torch.max(critic_loss_unclipped, critic_loss_clipped)
                    critic_loss = critic_loss_max.mean()
                else:
                    critic_loss = F.mse_loss(v_target[index], v_s)
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.ac.critic.parameters(), self.max_grad_norm)
                self.optimizer_critic.step()

                # self.optimizer.zero_grad()
                # loss = actor_loss.mean() + critic_loss * 0.5
                # loss.backward()
                # if self.use_grad_clip:  # Trick 7: Gradient clip
                #     torch.nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                # self.optimizer.step()

                # if self.args.target_kl is not None:
                #     if approx_kl > self.args.target_kl:
                #         break

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(cur_steps, total_steps)

    def lr_decay(self, cur_steps, total_steps):
        if cur_steps > total_steps:
            cur_steps = total_steps
        lr_a_now = 0.9 * self.lr_a * (1 - cur_steps / total_steps) + 0.1 * self.lr_a
        lr_c_now = 0.9 * self.lr_c * (1 - cur_steps / total_steps) + 0.1 * self.lr_c
        for p in self.optimizer_actor.param_groups:
            p["lr"] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p["lr"] = lr_c_now
