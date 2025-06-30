# agents/base_agent.py

from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def store_transition(self, *args):
        pass

    @abstractmethod
    def learn(self):
        pass