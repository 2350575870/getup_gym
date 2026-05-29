from abc import ABC, abstractmethod

class RunnerBase(ABC):

    "Base class for all runners"

    @abstractmethod
    def learn(self):

        "Method to train the algorithm"

        raise NotImplementedError

    @abstractmethod
    def log(self, locs: dict, width: int, pad: int) -> None:

        "Method to log the training process"

        #input: locs: dictionary of log information, 
        #       width: width of the log
        #       pad: padding of the log

        raise NotImplementedError

    @abstractmethod
    def save(self, path: str, info = None) -> None:

        "Method to save the model"

        #input: path: path to save the model, 
        #       info: additional information to save

        raise NotImplementedError

    @abstractmethod
    def load(self, path: str, load_optimizer: bool = True) -> dict:

        "Method to load the model"

        #input: path: path to load the model
        #       load_optimizer: whether to load the optimizer state

        raise NotImplementedError

    @abstractmethod
    def get_inference_policy(self):

        "Method to get the inference model"

        raise NotImplementedError
