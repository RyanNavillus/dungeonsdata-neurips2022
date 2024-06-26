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
from . import tasks
from . import wrappers
from syllabus.core import MultiProcessingSyncWrapper
from syllabus.examples.task_wrappers import NethackTaskWrapper
from nle import nethack


class GymConvWrapper():
    def __init__(self, env):
        self.env = env

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        return obs, rew, term or trunc, info

    def reset(self):
        obs, info = self.env.reset()
        return obs


def create_env(flags, curriculum=None, task_queue=None, update_queue=None):
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
        # "inv_glyphs",
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
    if flags.env.name == "challenge":
        kwargs.update(no_progress_timeout=150)
    else:
        kwargs.update(actions=nethack.ACTIONS)

    if flags.env.name in ("staircase", "pet", "oracle"):
        kwargs.update(reward_win=flags.reward_win, reward_lose=flags.reward_lose)
    # else:  # print warning once
    # warnings.warn("Ignoring flags.reward_win and flags.reward_lose")
    if flags.state_counter != "none":
        kwargs.update(state_counter=flags.state_counter)
    env = env_class(**kwargs)

    if flags.add_image_observation:
        env = wrappers.RenderCharImagesWithNumpyWrapperV2(
            env,
            crop_size=flags.crop_dim,
            rescale_font_size=(flags.pixel_size, flags.pixel_size),
        )

    if flags.syllabus:
        env = NethackTaskWrapper(env)

    if curriculum is not None:
        env = MultiProcessingSyncWrapper(
            env,
            curriculum.get_components(),
            update_on_step=False,
            task_space=env.task_space,
            buffer_size=1,
        )
        env = GymConvWrapper(env)
    # if task_queue is not None and update_queue is not None:
    #     env = MultiProcessingSyncWrapper(
    #         env,
    #         task_queue,
    #         update_queue,
    #         update_on_step=False,
    #         task_space=env.task_space,
    #     )

    return env
