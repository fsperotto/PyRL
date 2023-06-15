from gymnasium.envs.registration import register
from .survival import SurvivalEnv

register(
    id="Survival-v0",
    entry_point="pyrl.environments:SurvivalEnv",
)