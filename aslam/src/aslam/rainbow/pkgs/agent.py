#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import os
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from aslam.rainbow.pkgs.Model.model import DQN

class Agent():
    def __init__(self, args, env):
        self.action_space = env.action_space.n
        self.atoms = args.atoms
        self.Vmin = args.V_min
        self.Vmax = args.V_max
        self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)
        self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
        self.batch_size = args.batch_size
        self.n = args.multi_step
        self.discount = args.discount
        self.norm_clip = args.norm_clip
        self.device = args.device

        # Initialize online network
        self.online_net = DQN(args, self.action_space).to(device=args.device)
        if args.model and os.path.isfile(args.model):
            state_dict = torch.load(args.model, map_location='cpu', weights_only=True)
            self.online_net.load_state_dict(state_dict)
            print(f"Loading pretrained model: {args.model}")
        elif args.model:
            raise FileNotFoundError(args.model)
        self.online_net.train()

        # Initialize target network
        self.target_net = DQN(args, self.action_space).to(device=args.device)
        self.update_target_net()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False

        # Optimizer
        self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

    # Resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        self.online_net.reset_noise()

    # Acts based on single state (no batch)
    def act(self, state):
        torch_state = {}
        with torch.no_grad():
            for key in state.keys(): 
            	torch_state[key] = torch.tensor(state[key], dtype=torch.float32, device=self.device).unsqueeze(0)
            	
            return (self.online_net(torch_state) * self.support).sum(2).argmax(1).item()


    # Acts with an ε-greedy policy
    def act_e_greedy(self, state, epsilon=0.001):  # High ε can reduce evaluation scores drastically
        return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)


    def learn(self, mem):
        # Sample transitions
        idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)

        # Calculate current state probabilities (online network noise already sampled)
        log_ps = self.online_net(states, log=True)
        log_ps_a = log_ps[range(self.batch_size), actions]

        with torch.no_grad():
            # Calculate nth next state probabilities
            pns = self.online_net(next_states)   # Probabilities p(s_t+n, ·; θonline)
            dns = self.support.expand_as(pns) * pns # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
            argmax_indices_ns = dns.sum(2).argmax(1) # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            self.target_net.reset_noise() # Sample new target net noise
            pns = self.target_net(next_states) # Probabilities p(s_t+n, ·; θtarget)
            pns_a = pns[range(self.batch_size), argmax_indices_ns] # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

            # Compute Tz (Bellman operator T applied to z)
            Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0) # Tz = R^n + (γ^n)z (accounting for terminal states)
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax) # Clamp between supported values
            b = (Tz - self.Vmin) / self.delta_z # b = (Tz - Vmin) / Δz
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            m = torch.zeros(size=(self.batch_size, self.atoms), device=self.device)
            offset = torch.linspace(0, (self.batch_size - 1) * self.atoms, self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
            m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1)) # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1)) # m_u = m_u + p(s_t+n, a*)(b - l)

        loss = -torch.sum(m * log_ps_a, 1) # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
        self.online_net.zero_grad()
        (weights * loss).mean().backward() # Backpropagate importance-weighted minibatch loss
        clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
        self.optimiser.step()

        mem.update_priorities(idxs, loss.detach().cpu().numpy()) # Update priorities of sampled transitions


    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # Save model parameters on current device (don't move model between devices)
    def save(self, path, name='model.pth'):
        torch.save(self.online_net.state_dict(), os.path.join(path, name))

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()

