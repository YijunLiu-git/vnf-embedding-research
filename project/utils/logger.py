# utils/logger.py

import os
import csv

class Logger:
    def __init__(self, log_dir, agent_name="agent", filename="log.csv"):
        os.makedirs(log_dir, exist_ok=True)
        self.file_path = os.path.join(log_dir, filename)
        self.file = open(self.file_path, mode='w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(["episode", "agent", "reward", "success", "sar", "splat"])
        self.agent_name = agent_name

    def log(self, episode, reward, success, sar, splat):
        self.writer.writerow([episode, self.agent_name, reward, success, sar, splat])
        self.file.flush()

    def close(self):
        self.file.close()