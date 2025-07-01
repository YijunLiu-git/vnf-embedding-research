# agents/hrl_controller.py

from agents.base_agent import BaseAgent
from agents.ddqn_agent import DDQNAgent

class HRLController(BaseAgent):
    def __init__(self, high_level_config, low_level_config):
        self.high_agent = DDQNAgent(**high_level_config)
        self.low_agent = DDQNAgent(**low_level_config)

    def select_action(self, state):
        high_decision = self.high_agent.select_action(state)
        if high_decision == 0:
            return -1  # 表示拒绝嵌入请求
        else:
            return self.low_agent.select_action(state)

    def store_transition(self, state, action, reward, next_state, done):
        self.high_agent.store_transition(state, action, reward, next_state, done)
        self.low_agent.store_transition(state, action, reward, next_state, done)

    def learn(self):
        self.high_agent.learn()
        self.low_agent.learn()