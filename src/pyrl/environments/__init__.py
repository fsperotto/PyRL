from gymnasium.envs.registration import register
from .survival import SurvivalEnv
from .tensorforce_survival import CustomEnvironment

register(
    id="Survival-v0",
    entry_point="pyrl.environments:SurvivalEnv",
)