"""
src/algorithms/__init__.py
==========================
Exposes the REGISTRY and the ``get_algorithm`` factory.

Importing this package triggers the registration of all algorithm subclasses.
"""

from src.algorithms.base       import REGISTRY, BaseAlgorithm, ScalerBundle  # noqa: F401

# Import subclasses — side-effect: each registers itself in REGISTRY
from src.algorithms.pi_lstm           import PILSTMAlgorithm             # noqa: F401
from src.algorithms.neural_ode        import NeuralODEAlgorithm          # noqa: F401
from src.algorithms.cdae_pi_lstm      import CDAEPILSTMAlgorithm         # noqa: F401
from src.algorithms.cvae_pi_lstm      import CVAEPILSTMAlgorithm         # noqa: F401
from src.algorithms.pca_pi_lstm       import PCAPILSTMAlgorithm          # noqa: F401
from src.algorithms.pls_pi_lstm       import PLSPILSTMAlgorithm          # noqa: F401
from src.algorithms.delta_cdae_pi_lstm   import DeltaCDAEPILSTMAlgorithm    # noqa: F401
from src.algorithms.cdae_process_pi_lstm import CDAEProcessPILSTMAlgorithm  # noqa: F401
from src.algorithms.process_pi_lstm      import ProcessPILSTMAlgorithm      # noqa: F401


def get_algorithm(name: str) -> type:
    """
    Return the algorithm class for ``name``.

    Parameters
    ----------
    name : str
        One of: ``"pi_lstm"``, ``"neural_ode"``,
        ``"cdae_pi_lstm"``, ``"cvae_pi_lstm"``,
        ``"pca_pi_lstm"``, ``"pls_pi_lstm"``,
        ``"delta_cdae_pi_lstm"``, ``"cdae_process_pi_lstm"``,
        ``"process_pi_lstm"``.

    Raises
    ------
    KeyError if the name is not registered.
    """
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown algorithm '{name}'. "
            f"Available: {sorted(REGISTRY.keys())}"
        )
    return REGISTRY[name]


__all__ = [
    "REGISTRY",
    "BaseAlgorithm",
    "ScalerBundle",
    "PILSTMAlgorithm",
    "NeuralODEAlgorithm",
    "CDAEPILSTMAlgorithm",
    "CVAEPILSTMAlgorithm",
    "PCAPILSTMAlgorithm",
    "PLSPILSTMAlgorithm",
    "DeltaCDAEPILSTMAlgorithm",
    "CDAEProcessPILSTMAlgorithm",
    "ProcessPILSTMAlgorithm",
    "get_algorithm",
]
