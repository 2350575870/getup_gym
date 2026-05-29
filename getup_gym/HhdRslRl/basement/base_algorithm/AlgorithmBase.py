from abc import ABC, abstractmethod
from getup_gym.HhdRslRl.basement.base_modules.ModulesBase import ModulesBase
from getup_gym.HhdRslRl.basement.base_storage.StorageBase import StorageBase, TransitionBase
import torch

class AlgorithmBase(ABC):
    "Base class for all algorithms"

    #include params

    #modules class, modules can be policy, critic, encoder, decoder, etc
    modules: ModulesBase

    #storage class, storage can be replay buffer, trajectory buffer, etc
    storage: StorageBase

    #transition class, transition can be state, action, reward, next_state, done, etc
    transition: TransitionBase

    # #optim function
    # optimizer: torch.optim 

    def init_storage(self) -> None:

        """Initialize the storage"""

        raise NotImplementedError

    @abstractmethod
    def act(self) -> TransitionBase:

        """Get all of the params from the modules and return a transition"""

        raise NotImplementedError

    @abstractmethod
    def update(self):

        "Update the algorithm"

        raise NotImplementedError

    # @abstractmethod
    # def broadcast_parameters(self) -> None:

    #     """Broadcast the parameters of the modules to all the GPUs"""

    #     raise NotImplementedError

    # @abstractmethod
    # def reduce_parameters(self) -> None:

    #     """Collect gradients from all GPUs and average them.

    #     This function is called after the backward pass to synchronize the gradients across all GPUs.
    #     """

    #     raise NotImplementedError


