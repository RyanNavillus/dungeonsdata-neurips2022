# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gymnasium as gym
from nle import nethack
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
from syllabus.core import GymnasiumSyncWrapper, GymnasiumEvaluationWrapper
from syllabus.examples.task_wrappers import NethackTaskWrapper, NethackSeedWrapper

from . import tasks, wrappers


class GymConvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.task_space = self.env.task_space

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        return obs, rew, term or trunc, info

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs


def create_env(flags, curriculum=None, task_wrapper=False, eval=False):
    env_class = tasks.ENVS[flags.env.name]
    observation_keys = (
        "message",
        "blstats",
        "tty_chars",
        "tty_colors",
        "tty_cursor",
        # ALSO AVAILABLE (OFF for speed)
        # "specials",
        # "colors",
        # "chars",
        # "glyphs",
        "inv_glyphs",
        # "inv_strs",
        # "inv_letters",
        # "inv_oclasses",
    )
    kwargs = dict(
        savedir=None,
        character=flags.character,
        max_episode_steps=flags.env.max_episode_steps,
        observation_keys=observation_keys,
        penalty_step=flags.penalty_step,
        penalty_time=flags.penalty_time,
        penalty_mode=flags.fn_penalty_step,
        options=nethack.NETHACKOPTIONS,
    )
    if flags.env.name in ["seed", "challenge"]:
        kwargs.update(no_progress_timeout=150)
    else:
        kwargs.update(actions=nethack.ACTIONS)
    if flags.env.name in ("staircase", "pet", "oracle"):
        kwargs.update(reward_win=flags.reward_win, reward_lose=flags.reward_lose)
    if flags.state_counter != "none":
        kwargs.update(state_counter=flags.state_counter)
    env = env_class(**kwargs)

    if flags.add_image_observation:
        env = wrappers.RenderCharImagesWithNumpyWrapperV2(
            env,
            crop_size=flags.crop_dim,
            rescale_font_size=(flags.pixel_size, flags.pixel_size),
        )
    if eval:
        env = GymV21CompatibilityV0(env=env)
        env = NethackSeedWrapper(env, num_seeds=flags.num_seeds)
        env = GymnasiumEvaluationWrapper(env, task_space=env.task_space, randomize_order=False, start_index_spacing=1)
        # env = GymnasiumEvaluationWrapper(env, task_space=env.task_space, randomize_order=True, ignore_seed=True)

    elif flags.syllabus:
        if curriculum is not None or task_wrapper:
            env = GymV21CompatibilityV0(env=env)
            env = NethackSeedWrapper(env, num_seeds=flags.num_seeds)
            if curriculum is not None:
                env = GymnasiumSyncWrapper(
                    env,
                    env.task_space,
                    curriculum.components,
                    buffer_size=2,
                    batch_size=32,
                    remove_keys=['glyphs', 'chars', 'colors', 'specials', 'blstats', 'message',
                                 'inv_glyphs', 'inv_strs', 'inv_letters', 'inv_oclasses', 'misc'],
                )
            env = GymConvWrapper(env)
    return env
