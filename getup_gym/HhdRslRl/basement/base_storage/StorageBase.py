from abc import ABC, abstractmethod
import torch

class TransitionBase(ABC):

    "Base class for all transitions"

    @abstractmethod
    def clear(self):

        "Clear the transition"

        raise NotImplementedError


class StorageBase(ABC):

    "Base class for all storage systems"

    #mini batch step
    step: int = 0

    @abstractmethod
    def add_transitions(self, transition: TransitionBase):

        """Add a transition to the storage"""

        #input: transition: TransitionBase object

        raise NotImplementedError

    @abstractmethod
    def mini_batch_generator(self, num_mini_batchs: int, num_epochs: int) -> dict:

        """Generate mini-batches from the storage"""

        #input: batch_size: size of the mini-batch
        #output: generator of mini-batches

        raise NotImplementedError

    def clear(self):

        """Clear the step"""

        self.step = 0