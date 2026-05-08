# World Wide Web

A web navigation AI agent trained with reinforcement learning, using a learned world model as a simulator. The agent learns to complete tasks on real websites — like searching for information, filling out forms, and navigating multi-page workflows — without needing a live browser for every training step.

---

## Overview

Training web agents the traditional way is slow and fragile. Every action requires waiting on a real browser to load a page, and one network hiccup can crash a training run. This project takes a different approach:

1. **Learn a world model** — a smaller LLM that predicts how a webpage changes when an action is taken
2. **Train an RL agent against the world model** — the agent "dreams" simulated rollouts at GPU speed instead of waiting on browsers

The result is a navigation policy that can be trained orders of magnitude faster than browser-in-the-loop methods.
