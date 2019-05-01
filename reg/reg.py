from typing import Dict, List


class REG:
    def __init__(self, train_data, dev_data):
        pass
        # self.data = data

    def pre_process(self):  # Do any manipulations to the train and dev sets
        return None

    def train(self):  # Train your pre-processed files, save checkpoints, return model
        return None

    def generate(self, text: str, entities: Dict[str, List[str]]) -> str:
        raise NotImplementedError("Must implement relex")
