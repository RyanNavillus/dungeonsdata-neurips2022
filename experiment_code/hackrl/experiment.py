import concurrent
import copy
import dataclasses
import getpass
import logging
import math
import os
import pprint
import signal
import socket
import time
from typing import Optional

import coolname
import hackrl.environment
import hackrl.models
import hydra
import numpy as np
import omegaconf
import render_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from hackrl.core import nest, record, vtrace
from nle.dataset import dataset, db, populate_db
from nle.env.tasks import NetHackEat, NetHackGold, NetHackScore, NetHackScout
from syllabus.core import MoolibEvaluator, make_multiprocessing_curriculum, DummyEvaluator
from syllabus.curricula import (CentralPrioritizedLevelReplay,
                                DomainRandomization, Constant,
                                PrioritizedLevelReplay, SequentialCurriculum, DirectPrioritizedLevelReplay)
from syllabus.examples.task_wrappers import NetHackCollect, NetHackDescend

import moolib

# TTYREC_ASYNC_ITERATOR = None
# TTYREC_DATA = None
TTYREC_HIDDEN_STATE = None
TTYREC_ENVPOOL = None


class TtyrecEnvPool:
    def __init__(self, flags, **dataset_kwargs):
        self.idx = 0
        self.env_pool_size = flags.ttyrec_envpool_size
        self.dataset = dataset.TtyrecDataset(flags.dataset, **dataset_kwargs)
        self.dataset.shuffle = True
        self.threadpool = dataset_kwargs["threadpool"]

        env = hackrl.environment.create_env(flags)
        obs = env.reset()

        embed_actions = torch.zeros((256, 1))
        for i, a in enumerate(env.actions):
            embed_actions[a.value][0] = i
        self.embed_actions = torch.nn.Embedding.from_pretrained(embed_actions)
        self.embed_actions = self.embed_actions.to(flags.device)
        self.char_array = np.ascontiguousarray(env.char_array)
        self.crop_dim = flags.crop_dim

        self.device = flags.device
        self.dataset_warmup = flags.dataset_warmup
        self.ttyrec_batch_size = flags.ttyrec_batch_size
        self.ttyrec_unroll_length = flags.ttyrec_unroll_length
        self.prev_action_shape = (flags.ttyrec_batch_size, flags.ttyrec_unroll_length)
        self.screen_shape = self.prev_action_shape + obs["screen_image"].shape

        self._iterators = []
        self._results = []
        self.bootstrap_actions = flags.dataset_bootstrap_actions
        self.bootstrap_pred_max = flags.bootstrap_pred_max
        if self.bootstrap_actions:
            self.inverse_model = self.load_bootstrap_model(flags.dataset_bootstrap_path)

        for _ in range(self.env_pool_size):
            it = self.make_single_iter(self.dataset)
            self._iterators.append(it)
            self._results.append(self.threadpool.submit(next, it))

    def load_bootstrap_model(self, path):
        load_data = torch.load(path)
        flags = omegaconf.OmegaConf.create(load_data["flags"])
        inverse_model = hackrl.models.create_model(flags, FLAGS.device)
        inverse_model.load_state_dict(load_data["learner_state"]["model"])
        return inverse_model

    def result(self):
        return self._results[self.idx].result()

    def step(self):
        fut = self.threadpool.submit(next, self._iterators[self.idx])
        self._results[self.idx] = fut
        self.idx = (self.idx + 1) % self.env_pool_size

    def make_single_iter(self, dataset):
        def _iter():
            mb_tensors = {
                "screen_image": torch.zeros(self.screen_shape, dtype=torch.uint8),
                "prev_action": torch.zeros(self.prev_action_shape, dtype=torch.uint8),
            }

            prev_action = torch.zeros(
                (self.ttyrec_batch_size, 1), dtype=torch.uint8
            ).to(self.device)
            while True:
                for i, mb in enumerate(dataset):

                    if i == 0:
                        # create torch tensors from first minibatch
                        screen_image = mb_tensors["screen_image"].numpy()
                        for k, array in mb.items():
                            mb_tensors[k] = torch.from_numpy(array)
                        [v.pin_memory() for v in mb_tensors.values()]

                    if i < self.dataset_warmup:
                        continue

                    cursor_uint8 = mb["tty_cursor"].astype(np.uint8)

                    def convert(i): return render_utils.render_crop(
                        mb["tty_chars"][i],
                        mb["tty_colors"][i],
                        cursor_uint8[i],
                        self.char_array,
                        screen_image[i],
                        self.crop_dim,
                    )
                    list(self.threadpool.map(convert, range(self.ttyrec_batch_size)))

                    final_mb = {
                        "tty_chars": mb_tensors["tty_chars"],
                        "tty_colors": mb_tensors["tty_colors"],
                        "tty_cursor": torch.from_numpy(cursor_uint8),
                        "screen_image": mb_tensors["screen_image"],
                        "done": mb_tensors["done"].bool(),
                    }

                    if "actions" in mb_tensors:
                        actions = mb_tensors["actions"].long().to(self.device)
                        actions_converted = (
                            self.embed_actions(actions).squeeze(-1).long()
                        )
                        final_mb["score"] = mb_tensors["scores"]
                        final_mb["actions_converted"] = actions_converted
                        final_mb["prev_action"] = torch.cat(
                            [prev_action, actions_converted[:, :-1]], dim=1
                        )
                        prev_action = actions_converted[:, -1:]

                    # DATASET is: [B T ...] but MODEL expects [T B ...]
                    yield {
                        k: t.transpose(0, 1).to(self.device)
                        for k, t in final_mb.items()
                    }

        def _bootstrap(generator):
            prev_hs = self.inverse_model.initial_state(self.ttyrec_batch_size)
            prev_hs = nest.map(lambda x: x.to(self.device), prev_hs)
            prev_state = None
            for mb in generator:

                if prev_state is None:
                    aug_mb = mb
                else:
                    aug_mb = {p: torch.cat([prev_state[p], mb[p]], dim=0) for p in mb}

                upto_last = nest.map(lambda x: x[:-1], aug_mb)
                last = nest.map(lambda x: x[-1:], aug_mb)

                with torch.no_grad():
                    upto_outputs, prev_hs = self.inverse_model(upto_last, prev_hs)
                    last_outputs, _ = self.inverse_model(last, prev_hs)

                outputs = {
                    p: torch.cat([upto_outputs[p], last_outputs[p]], dim=0)
                    for p in upto_outputs
                }

                T_plus1, B, *_ = outputs["inverse_action_logits"].shape
                if self.bootstrap_pred_max:
                    actions = torch.argmax(
                        outputs["inverse_action_logits"], dim=2
                    ).long()
                else:
                    actions = torch.multinomial(
                        F.softmax(
                            outputs["inverse_action_logits"].view(T_plus1 * B, -1),
                            dim=1,
                        ),
                        num_samples=1,
                    ).long()
                actions = actions.view(T_plus1, B)
                # actions = torch.argmax(outputs["inverse_action_logits"], dim=2).long()

                if prev_state is None:
                    mb["prev_action"] = torch.cat([actions[0:1], actions[:-1]], dim=0)
                    mb["actions_converted"] = actions
                    mb["inverse_action_logits"] = outputs["inverse_action_logits"]
                else:
                    mb["prev_action"] = actions[:-1]
                    mb["actions_converted"] = actions[1:]
                    mb["inverse_action_logits"] = outputs["inverse_action_logits"][1:]

                prev_state = last

                yield mb

        if self.bootstrap_actions:
            return iter(_bootstrap(_iter()))
        else:
            return iter(_iter())


def make_ttyrec_envpool(threadpool, flags):
    dbfilename = "/private/home/ehambro/fair/workspaces/clean_rl/ttyrecs.db"

    if not os.path.isfile(dbfilename):
        alt_path = "/scratch/ehambro/altorg/altorg/111720"
        aa_path = "/private/home/ehambro/fair/workspaces/autoascend-submission/nle_data"
        db.create(dbfilename)
        populate_db.add_nledata_directory(aa_path, "autoascend", dbfilename)
        populate_db.add_altorg_directory(alt_path, "altorg", dbfilename)

    kwargs = dict(
        batch_size=flags.ttyrec_batch_size,
        seq_length=flags.ttyrec_unroll_length,
        dbfilename=dbfilename,
        threadpool=threadpool,
        loop_forever=True,
        shuffle=True,
    )
    subselect = []
    if flags.character == "mon-hum-neu-mal":
        subselect.append(" role='Mon' AND race='Hum' ")
    if flags.dataset_demigod:
        subselect.append(" death='ascended' ")
    if flags.dataset_highscore:
        subselect.append(" points>10000")
    if flags.dataset_midscore:
        subselect.append(" points>1000 AND points<10000")

    if subselect:
        kwargs["subselect_sql"] = "SELECT gameid FROM games WHERE " + "AND".join(
            subselect
        )

    return TtyrecEnvPool(flags, **kwargs)


@dataclasses.dataclass
class StatMean:
    # Compute using Welford'd Online Algorithm
    # Algo: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    # Math: https://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/
    n: int = 0
    mu: float = 0
    m2: float = 0
    cumulative: bool = False

    def result(self):
        if self.n == 0:
            return None
        return self.mu

    def mean(self):
        return self.mu

    def std(self):
        if self.n < 1:
            return None
        return math.sqrt(self.m2 / self.n)

    def __sub__(self, other):
        assert isinstance(other, StatMean)
        n_new = self.n - other.n
        if n_new == 0:
            return StatMean(0, 0, 0)
        mu_new = (self.mu * self.n - other.mu * other.n) / n_new
        delta = other.mu - mu_new
        m2_new = self.m2 - other.m2 - (delta**2) * n_new * other.n / self.n
        return StatMean(n_new, mu_new, m2_new)

    def __iadd__(self, other):
        if isinstance(other, StatMean):
            other_n = other.n
            other_mu = other.mu
            other_m2 = other.m2
        elif isinstance(other, torch.Tensor):
            other_n = other.numel()
            other_mu = other.mean().item()
            other_m2 = ((other - other_mu) ** 2).sum().item()
        else:
            other_n = 1
            other_mu = other
            other_m2 = 0
        # See parallelized Welford in wiki
        new_n = other_n + self.n
        delta = other_mu - self.mu
        self.mu += delta * (other_n / max(new_n, 1))
        delta2 = other_mu - self.mu
        self.m2 += other_m2 + (delta2**2) * (self.n * other_n / max(new_n, 1))
        self.n = new_n
        return self

    def reset(self):
        if not self.cumulative:
            self.mu = 0
            self.n = 0

    def decay_cumulative(self, n=1e6):
        """Adjust sample size downwards to upweight recent samples"""
        if not self.cumulative:
            return
        if self.n > n:
            self.m2 *= n / self.n
            self.n = n

    def __repr__(self):
        return repr(self.result())


@dataclasses.dataclass
class StatSum:
    value: float = 0

    def result(self):
        return self.value

    def __sub__(self, other):
        assert isinstance(other, StatSum)
        return StatSum(self.value - other.value)

    def __iadd__(self, other):
        if isinstance(other, StatSum):
            self.value += other.value
        else:
            self.value += other
        return self

    def reset(self):
        pass

    def decay_cumulative(self):
        pass

    def __repr__(self):
        return repr(self.result())


@dataclasses.dataclass
class StatMax:
    value: float = 0

    def result(self):
        return self.value

    def __sub__(self, other):
        assert isinstance(other, StatMax)
        return StatMax(self.value - other.value)

    def __iadd__(self, other):
        if isinstance(other, StatMax):
            self.value = max(self.value, other.value)
        else:
            self.value = max(self.value, other)
        return self

    def reset(self):
        pass

    def decay_cumulative(self):
        pass

    def __repr__(self):
        return repr(self.result())


@dataclasses.dataclass
class LearnerState:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    model_version: int = 0
    num_previous_leaders: int = 0
    train_time: float = 0
    last_checkpoint: float = 0
    last_checkpoint_history: float = 0
    global_stats: Optional[dict] = None

    def save(self):
        r = dataclasses.asdict(self)
        r["model"] = self.model.state_dict()
        r["optimizer"] = self.optimizer.state_dict()
        return r

    def load(self, state):
        for k, v in state.items():
            if k not in ("model", "optimizer", "global_stats"):
                setattr(self, k, v)
        self.model.version = state["model_version"]
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])

        for k, v in state["global_stats"].items():
            if k in self.global_stats:
                self.global_stats[k] = type(self.global_stats[k])(**v)


class GlobalStatsAccumulator:
    """Class for global accumulation state. add_stats gets reduced."""

    def __init__(self, rpc_group, global_stats):
        self.rpc_group = rpc_group
        self.global_stats = global_stats

        self.reduce_future = None
        self.queued_global_stats = None
        self.sent_global_stats = None
        self.prev_stats = None

    def add_stats(self, dst, src):
        for k, v in dst.items():
            v += src[k]
        return dst

    def enqueue_global_stats(self, stats):
        if self.queued_global_stats is None:
            self.queued_global_stats = copy.deepcopy(stats)
        else:
            # Sum pending data.
            self.add_stats(self.queued_global_stats, stats)

    def reduce(self, stats):
        if self.reduce_future is not None and self.reduce_future.done():
            if self.reduce_future.exception() is not None:
                logging.info(
                    "global stats accumulation error: %s",
                    self.reduce_future.exception(),
                )
                self.enqueue_global_stats(self.sent_global_stats)
            else:
                self.add_stats(self.global_stats, self.reduce_future.result())
                for v in self.global_stats.values():
                    v.decay_cumulative()
            self.reduce_future = None

        for v in stats.values():
            v.decay_cumulative()

        stats_diff = stats
        if self.prev_stats is not None:
            stats_diff = {k: v - self.prev_stats[k] for k, v in stats.items()}

        self.enqueue_global_stats(stats_diff)
        self.prev_stats = copy.deepcopy(stats)

        if self.reduce_future is None:
            # Only reduce when not currently reducing.
            # Otherwise, we keep queued_global_stats for next time.
            self.sent_global_stats = self.queued_global_stats
            self.queued_global_stats = None
            # Additional copy to deal with potential partial reductions.
            self.reduce_future = self.rpc_group.all_reduce(
                "global stats", copy.deepcopy(self.sent_global_stats), op=self.add_stats
            )

    def reset(self):
        if self.prev_stats is not None:
            for _, v in self.prev_stats.items():
                v.reset()


class EnvBatchState:
    def __init__(self, flags, model, actor_batch_size=None):
        batch_size = actor_batch_size if actor_batch_size is not None else flags.actor_batch_size
        device = flags.device
        self.batch_size = batch_size
        self.prev_action = torch.zeros(batch_size).long().to(device)
        self.future = None
        self.core_state = model.initial_state(batch_size=batch_size)
        self.core_state = nest.map(lambda x: x.to(device), self.core_state)
        self.initial_core_state = self.core_state
        self.discount = flags.discounting

        self.running_reward = torch.zeros(batch_size)
        self.discounted_running_reward = torch.zeros(batch_size)
        self.step_count = torch.zeros(batch_size)

        self.time_batcher = moolib.Batcher(flags.unroll_length + 1, flags.device)
        self.prev_blstats = None

    def update(self, env_outputs, action, stats):
        self.prev_action = action
        self.running_reward += env_outputs["reward"]
        self.discounted_running_reward *= self.discount
        self.discounted_running_reward += env_outputs["reward"]
        self.step_count += 1

        done = env_outputs["done"]

        episode_return = self.running_reward * done
        episode_step = self.step_count * done
        episodes_done = done.sum().item()

        if episodes_done > 0:
            stats["mean_episode_return"] += episode_return.sum().item() / episodes_done
            stats["mean_episode_step"] += episode_step.sum().item() / episodes_done
        stats["steps_done"] += done.numel()
        stats["episodes_done"] += episodes_done

        stats["running_reward"] += self.running_reward.mean().item()
        stats["discounted_running_reward"] += self.discounted_running_reward.mean().item()
        stats["running_step"] += self.step_count.mean().item()

        stats["mean_square_discounted_running_reward"] += self.discounted_running_reward ** 2
        not_done = ~done

        self.discounted_running_reward *= not_done
        self.running_reward *= not_done
        self.step_count *= not_done

        # Track end game stats
        done_count = torch.sum(env_outputs["done"]).item()
        # Done signal comes with the first step of the next episode, so we use the previous blstats
        if self.prev_blstats is not None:
            # Zero all non-finished episode values
            dungeon_level = self.prev_blstats[:, 12]
            character_level = self.prev_blstats[:, 18]
            gold = self.prev_blstats[:, 13]
            final_dungeon_level = torch.where(
                env_outputs["done"], dungeon_level, torch.zeros_like(dungeon_level))
            final_character_level = torch.where(
                env_outputs["done"], character_level, torch.zeros_like(character_level))
            final_gold = torch.where(
                env_outputs["done"], gold, torch.zeros_like(gold))
            if done_count > 0:
                # Only log stats if there is at least one value to record
                mean_final_dungeon_level = final_dungeon_level.float().sum().item() / float(done_count)
                mean_final_character_level = final_character_level.float().sum().item() / float(done_count)
                mean_final_gold = final_gold.float().sum().item() / float(done_count)
                max_final_dungeon_level = final_dungeon_level.float().max().item()
                max_final_character_level = final_character_level.float().max().item()
                max_final_gold = final_gold.float().max().item()
                stats["mean_final_dungeon_level"] += mean_final_dungeon_level
                stats["mean_final_character_level"] += mean_final_character_level
                stats["mean_final_gold"] += mean_final_gold
                stats["max_final_dungeon_level"] += max_final_dungeon_level
                stats["max_final_character_level"] += max_final_character_level
                stats["max_final_gold"] += max_final_gold
        self.prev_blstats = env_outputs["blstats"].clone()


def compute_baseline_loss(
    actor_baseline, learner_baseline, target, clip_delta_value=None, stats=None, per_item=False
):
    baseline_loss = (target - learner_baseline) ** 2

    if clip_delta_value:
        # Common PPO trick - clip a change in baseline fn
        # (cf PPO2 github.com/Stable-Baselines-Team/stable-baselines)
        delta_baseline = learner_baseline - actor_baseline
        clipped_baseline = actor_baseline + torch.clamp(
            delta_baseline, -clip_delta_value, clip_delta_value
        )

        clipped_baseline_loss = (target - clipped_baseline) ** 2

        if stats:
            clipped = (clipped_baseline_loss > baseline_loss).float().mean().item()
            stats["clipped_baseline_fraction"] += clipped

        baseline_loss = torch.max(baseline_loss, clipped_baseline_loss)

    if stats:
        stats["max_baseline_value"] += torch.max(learner_baseline).item()
        stats["min_baseline_value"] += torch.min(learner_baseline).item()
        stats["mean_baseline_value"] += torch.mean(learner_baseline).item()

    if per_item:
        return baseline_loss

    return 0.5 * torch.mean(baseline_loss)


def compute_entropy_loss(logits, stats=None):
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    entropy_per_timestep = torch.sum(-policy * log_policy, dim=-1)
    if stats:
        stats["max_entropy_value"] += torch.max(entropy_per_timestep).item()
        stats["min_entropy_value"] += torch.min(entropy_per_timestep).item()
        stats["mean_entropy_value"] += torch.mean(entropy_per_timestep).item()
    return -torch.mean(entropy_per_timestep)


def compute_kickstarting_loss(student_logits, expert_logits):
    T, B, *_ = student_logits.shape
    return torch.nn.functional.kl_div(
        F.log_softmax(student_logits.view(T * B, -1), dim=-1),
        F.log_softmax(expert_logits.view(T * B, -1), dim=-1),
        log_target=True,
        reduction="batchmean",
    )


def compute_policy_gradient_loss(
    actor_log_prob,
    learner_log_prob,
    advantages,
    normalize_advantages=False,
    clip_delta_policy=None,
    stats=None,
):
    advantages = advantages.detach()
    stats["running_advantages"] += advantages

    adv = advantages

    if normalize_advantages:
        # Common PPO trick (cf PPO2 github.com/Stable-Baselines-Team/stable-baselines)
        if FLAGS.use_global_advantage_norm:
            sample_adv = stats["running_advantages"]
        else:
            sample_adv = adv
        advantages = (adv - sample_adv.mean()) / max(1e-3, sample_adv.std())
        stats["sample_advantages"] += advantages.mean().item()

    if clip_delta_policy:
        # APPO policy loss - clip a change in policy fn
        ratio = torch.exp(learner_log_prob - actor_log_prob)
        policy_loss = ratio * advantages

        clip_high = 1 + clip_delta_policy
        clip_low = 1.0 / clip_high

        clipped_ratio = torch.clamp(ratio, clip_low, clip_high)
        clipped_policy_loss = clipped_ratio * advantages

        if stats:
            clipped_fraction = (clipped_policy_loss < policy_loss).float().mean().item()
            stats["clipped_policy_fraction"] += clipped_fraction
        policy_loss = torch.min(policy_loss, clipped_policy_loss)
    else:
        # IMPALA policy loss
        policy_loss = learner_log_prob * advantages

    return -torch.mean(policy_loss)


def compute_inverse_loss(predicted_action_logits, actions):
    T, B = actions.shape
    return torch.nn.functional.cross_entropy(
        predicted_action_logits.reshape(T * B, -1), actions.reshape(T * B)
    )


def create_optimizer(model):
    return torch.optim.Adam(
        model.parameters(),
        lr=FLAGS.adam_learning_rate,
        betas=(FLAGS.adam_beta1, FLAGS.adam_beta2),
        eps=FLAGS.adam_eps,
    )


def compute_gradients(data, learner_state, stats, curriculum, actor_index=None):
    global TTYREC_ENVPOOL, TTYREC_HIDDEN_STATE
    model = learner_state.model

    env_outputs = data["env_outputs"]
    actor_outputs = data["actor_outputs"]
    initial_core_state = data["initial_core_state"]

    model.train()

    total_loss = 0

    if actor_index is None:
        actor_index = 0

    if FLAGS.supervised_loss or FLAGS.behavioural_clone:
        ttyrec_data = TTYREC_ENVPOOL.result()
        idx = TTYREC_ENVPOOL.idx
        ttyrec_predictions, TTYREC_HIDDEN_STATE[idx] = model(
            ttyrec_data, TTYREC_HIDDEN_STATE[idx]
        )
        TTYREC_HIDDEN_STATE[idx] = nest.map(
            lambda t: t.detach(), TTYREC_HIDDEN_STATE[idx]
        )

        true_a = torch.flatten(ttyrec_data["actions_converted"], 0, 1)
        logits = torch.flatten(ttyrec_predictions["policy_logits"], 0, 1)
        if FLAGS.bootstrap_is_kl:
            expert = torch.flatten(ttyrec_data["inverse_action_logits"], 0, 1)
            supervised_loss = FLAGS.supervised_loss * compute_kickstarting_loss(
                logits[:-1], expert[:-1]
            )
        else:
            supervised_loss = (
                FLAGS.supervised_loss * F.cross_entropy(logits[:-1], true_a[:-1]).mean()
            )
        stats["supervised_loss"] += supervised_loss.item()

        total_loss += supervised_loss
        if FLAGS.use_inverse_model:
            if FLAGS.use_inverse_model_only:
                total_loss = 0
            inverse_loss = FLAGS.inverse_loss * compute_inverse_loss(
                ttyrec_predictions["inverse_action_logits"][:-1],
                ttyrec_data["actions_converted"][:-1].long(),
            )
            total_loss += inverse_loss
            if FLAGS.augment_inverse_random:
                learner_outputs, _ = model(env_outputs, initial_core_state)
                random_inverse_loss = FLAGS.random_inverse_loss * compute_inverse_loss(
                    learner_outputs["inverse_action_logits"][:-1],
                    actor_outputs["action"][:-1].long(),
                )
                pred_a = torch.argmax(learner_outputs["inverse_action_logits"], dim=2)
                acc = (pred_a == actor_outputs["action"]).float().mean()
                stats["random_inverse_prediction_accuracy"] += acc.item()
                stats["random_inverse_loss"] += inverse_loss.item()
                total_loss += random_inverse_loss

            pred_a = torch.argmax(ttyrec_predictions["inverse_action_logits"], dim=2)
            acc = (pred_a == ttyrec_data["actions_converted"]).float().mean()
            stats["inverse_prediction_accuracy"] += acc.item()
            stats["inverse_loss"] += inverse_loss.item()

        # Only call step when you are done with ttyrec_data - it may get overwritten
        TTYREC_ENVPOOL.step()
        if FLAGS.behavioural_clone or FLAGS.use_inverse_model_only:
            stats["env_train_steps"] += (
                FLAGS.ttyrec_unroll_length * FLAGS.ttyrec_batch_size
            )
            total_loss.backward()
            return

    learner_outputs, _ = model(env_outputs, initial_core_state, get_action=False, get_logits=True)

    # Use last baseline value (from the value function) to bootstrap.
    bootstrap_value = learner_outputs["baseline"][-1]

    # Move from env_outputs[t] -> action[t] to action[t] -> env_outputs[t].
    learner_outputs = nest.map(lambda t: t[:-1], learner_outputs)
    env_outputs = nest.map(lambda t: t[1:], env_outputs)
    actor_outputs = nest.map(lambda t: t[:-1], actor_outputs)

    rewards = env_outputs["reward"] * FLAGS.reward_scale
    if FLAGS.rms_reward_norm:
        if not FLAGS.per_task_reward_norm or curriculum is None:
            reward_std = stats["mean_square_discounted_running_reward"].mean() ** 0.5
            reward_scale = max(0.01, reward_std)
        else:
            # We're cheating by passing tasks through the cursor since we can't pass info dicts in moolib
            tasks = env_outputs["tty_cursor"]
            stds = {t: curriculum.curriculum.stat_recorder.episode_returns[t].std(
            ) for t in range(curriculum.num_tasks)}

            # Check for invalid std dev values
            for k, v in stds.items():
                if v is None or math.isnan(v):
                    stds[k] = stats["mean_square_discounted_running_reward"].mean() ** 0.5

            reward_std = tasks.cpu().apply_(lambda t: stds[int(t)])
            reward_scale = torch.clamp(reward_std, min=0.01)

        rewards /= reward_scale
        stats["reward_scale"] += reward_scale
        stats["reward_normalised"] += rewards.mean().item()
    if FLAGS.reward_clip:
        rewards = torch.clip(rewards, -FLAGS.reward_clip, FLAGS.reward_clip)

    # if FLAGS.normalize_reward:
    #     # Only NetHackNet models
    #     model.update_running_moments(rewards)
    #     rewards /= model.get_running_std()

    discounts = (~env_outputs["done"]).float() * FLAGS.discounting

    vtrace_returns = vtrace.from_logits(
        behavior_policy_logits=actor_outputs["policy_logits"],
        target_policy_logits=learner_outputs["policy_logits"],
        actions=actor_outputs["action"],
        discounts=discounts,
        rewards=rewards,
        values=learner_outputs["baseline"],
        bootstrap_value=bootstrap_value,
    )

    entropy_loss = FLAGS.entropy_cost * compute_entropy_loss(
        learner_outputs["policy_logits"], stats
    )

    pg_loss = compute_policy_gradient_loss(
        vtrace_returns.behavior_action_log_probs,
        vtrace_returns.target_action_log_probs,
        vtrace_returns.pg_advantages,
        FLAGS.normalize_advantages,
        FLAGS.appo_clip_policy,
        stats,
    )

    baseline_loss = FLAGS.baseline_cost * compute_baseline_loss(
        actor_outputs["baseline"],
        learner_outputs["baseline"],
        vtrace_returns.vs,
        FLAGS.appo_clip_baseline,
        stats,
        per_item=False,
    )

    # Syllabus curriculum update
    if FLAGS.syllabus and FLAGS.curriculum_method == "simpleplr":
        current_tasks = env_outputs["tty_cursor"]
        scores = (vtrace_returns.vs - learner_outputs["baseline"]).abs()
        current_dones = env_outputs["done"]
        curriculum.update(current_tasks, scores, current_dones,
                          np.arange(actor_index, actor_index + FLAGS.batch_size))

    # Not sure if this is correct for every config
    actor_index = (actor_index + FLAGS.actor_batch_size) % (FLAGS.actor_batch_size * FLAGS.num_actor_batches)

    total_loss += entropy_loss + pg_loss + baseline_loss

    if FLAGS.use_inverse_model:
        inverse_loss = FLAGS.inverse_loss * compute_inverse_loss(
            learner_outputs["inverse_action_logits"], actor_outputs["action"].long()
        )
        if FLAGS.use_inverse_model_only:
            total_loss = 0
        total_loss += inverse_loss

        pred_a = torch.argmax(learner_outputs["inverse_action_logits"], dim=2)
        acc = (pred_a == actor_outputs["action"]).float().mean()
        stats["inverse_prediction_accuracy"] += acc.item()
        stats["inverse_loss"] += inverse_loss.item()

    if FLAGS.use_kickstarting:
        kickstarting_loss = FLAGS.kickstarting_loss * compute_kickstarting_loss(
            learner_outputs["policy_logits"],
            actor_outputs["kick_policy_logits"],
        )
        total_loss += kickstarting_loss
        stats["kickstarting_loss"] += kickstarting_loss.item()

    total_loss.backward()

    stats["env_train_steps"] += FLAGS.unroll_length * FLAGS.batch_size
    stats["policy_loss"] += pg_loss.item()
    stats["baseline_loss"] += baseline_loss.item()
    stats["entropy_loss"] += entropy_loss.item()

    policy_lag = model.version - actor_outputs["version"]
    stats["max_policy_lag"] += policy_lag.max().item()
    stats["mean_policy_lag"] += policy_lag.float().mean().item()
    stats["min_policy_lag"] += policy_lag.min().item()


def step_optimizer(learner_state, stats):
    optimizer = learner_state.optimizer
    model = learner_state.model

    unclipped_grad_norm = nn.utils.clip_grad_norm_(
        model.parameters(), FLAGS.grad_norm_clipping
    )
    optimizer.step()

    learner_state.model_version += 1
    learner_state.model.version += 1

    stats["unclipped_grad_norm"] += unclipped_grad_norm.item()
    stats["optimizer_steps"] += 1


def log(stats, step, is_global=False, is_eval=False, curriculum=None, allowlist=None):
    allowlist = stats.keys() if allowlist is None else allowlist
    stats_values = {}
    prefix = "global/" if is_global else "local/"
    prefix = "eval/" if is_eval else prefix
    for k, v in stats.items():
        if (is_global or is_eval) and k in allowlist:     # Reduce logging file size
            stats_values[prefix + k] = v.result()
        v.reset()

    if is_global:
        mer = stats_values["global/mean_episode_return"]
        logging.info("Mean episode Return %g", mer if mer is not None else 0)
    # logging.info(stats_values)
    # if not is_global:
    #     record.log_to_file(**stats_values)

    if FLAGS.wandb:
        stats_values["global_step"] = step
        wandb.log(stats_values)
        if curriculum is not None and is_global:
            curriculum.log_metrics(wandb, [], step=step + 1)


def save_checkpoint(checkpoint_path, learner_state):
    tmp_path = "%s.tmp.%s" % (checkpoint_path, moolib.create_uid())

    logging.info("saving global stats %s", learner_state.global_stats)

    checkpoint = {
        "learner_state": learner_state.save(),
        "flags": omegaconf.OmegaConf.to_container(FLAGS),
    }

    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, checkpoint_path)

    logging.info("Checkpoint saved to %s", checkpoint_path)


def load_checkpoint(checkpoint_path, learner_state):
    checkpoint = torch.load(checkpoint_path)
    learner_state.load(checkpoint["learner_state"])


def calculate_sps(stats, delta, prev_steps):
    env_train_steps = stats["env_train_steps"].result()
    stats["SPS"] += (env_train_steps - prev_steps) / delta
    logging.info("calculate_sps %g steps in %g: %g", env_train_steps -
                 prev_steps, delta, (env_train_steps - prev_steps) / delta)
    return env_train_steps


def uid():
    return "%s:%i:%s" % (socket.gethostname(), os.getpid(), coolname.generate_slug(2))


def evaluate_agent(FLAGS, model, eval_envs, eval_env_states, eval_stat_dict, next_eval_env_index=0):
    model.eval()
    initial_episodes_done = eval_stat_dict["episodes_done"].value
    print("Evaluating agent")
    start_time = time.time()
    while eval_stat_dict["episodes_done"].value < initial_episodes_done + FLAGS.eval_episodes:
        # Generate data.
        cur_eval_index = next_eval_env_index
        next_eval_env_index = (next_eval_env_index + 1) % FLAGS.num_actor_batches

        eval_env_state = eval_env_states[cur_eval_index]
        if eval_env_state.future is None:
            eval_env_state.future = eval_envs.step(cur_eval_index, eval_env_state.prev_action)
        cpu_eval_env_outputs = eval_env_state.future.result()

        eval_env_outputs = nest.map(
            lambda t: t.to(FLAGS.device, copy=True), cpu_eval_env_outputs
        )

        eval_env_outputs["prev_action"] = eval_env_state.prev_action
        prev_eval_core_state = eval_env_state.core_state
        model.eval()
        with torch.no_grad():
            eval_actor_outputs, eval_env_state.core_state = model(
                nest.map(lambda t: t.unsqueeze(0), eval_env_outputs),
                eval_env_state.core_state, get_value=False
            )
        eval_actor_outputs = nest.map(lambda t: t.squeeze(0), eval_actor_outputs)
        eval_action = eval_actor_outputs["action"]
        eval_env_state.update(cpu_eval_env_outputs, eval_action, eval_stat_dict)
        eval_env_state.future = eval_envs.step(cur_eval_index, eval_action)
        last_data = {
            "env_outputs": eval_env_outputs,
            "actor_outputs": eval_actor_outputs,
        }
        eval_env_state.time_batcher.stack(last_data)

        if not eval_env_state.time_batcher.empty():
            data = eval_env_state.time_batcher.get()
            data["initial_core_state"] = eval_env_state.initial_core_state

            # We need the last entry of the previous time batch
            # to be put into the first entry of this time batch,
            # with the initial_core_state to match
            eval_env_state.initial_core_state = prev_eval_core_state
            eval_env_state.time_batcher.stack(last_data)
    end_time = time.time()
    print("Evaluation took", end_time - start_time, "seconds")
    return eval_env_states, next_eval_env_index


def setup_curriculum(FLAGS, model=None):
    sample_env = hackrl.environment.create_env(FLAGS, task_wrapper=True)

    def task_names(task, idx): return task

    if FLAGS.curriculum_method == "sq":
        print("Using Sequential Curriculum")
        curriculum = SequentialCurriculum([
            [NetHackScout, NetHackDescend, NetHackScore],
            [NetHackGold, NetHackDescend, NetHackEat, NetHackScore],
            NetHackScore
        ], ["steps>=200000000", "steps>=500000000"], sample_env.task_space, record_stats=True, task_names=task_names)
    elif FLAGS.curriculum_method == "dr":
        print("Using Domain Randomization")
        curriculum = DomainRandomization(sample_env.task_space, record_stats=True, task_names=task_names)
    elif FLAGS.curriculum_method == "plr":
        print("Using Prioritized Level Replay")
        evaluator = MoolibEvaluator(model, device="cuda", copy_agent=True)
        # evaluator = DummyEvaluator(sample_env.action_space)
        curriculum = PrioritizedLevelReplay(
            sample_env.task_space,
            sample_env.observation_space,
            num_steps=FLAGS.batch_size,
            num_processes=FLAGS.actor_batch_size * FLAGS.num_actor_batches,
            num_minibatches=1,
            gamma=FLAGS.discounting,
            gae_lambda=0.95,
            task_sampler_kwargs_dict={"strategy": FLAGS.plr.strategy, "alpha": FLAGS.plr.alpha,
                                      "temperature": FLAGS.plr.temperature, "staleness_coef": FLAGS.plr.staleness_coef},
            evaluator=evaluator,
            lstm_size=FLAGS.baseline.hidden_dim,
            record_stats=True,
            task_names=task_names,
            device="cuda",
        )
    elif FLAGS.curriculum_method == "centralplr":
        print("Using Central Prioritized Level Replay")
        curriculum = CentralPrioritizedLevelReplay(
            sample_env.task_space,
            num_steps=FLAGS.batch_size,
            num_processes=FLAGS.actor_batch_size,
            gamma=FLAGS.discounting,
            gae_lambda=0.95,
            task_sampler_kwargs_dict={"strategy": "value_l1"},
            record_stats=True,
            task_names=task_names
        )
    elif FLAGS.curriculum_method == "simpleplr":
        print("Using Simple Prioritized Level Replay")
        curriculum = DirectPrioritizedLevelReplay(
            sample_env.task_space,
            num_steps=FLAGS.batch_size,
            num_processes=FLAGS.actor_batch_size * FLAGS.num_actor_batches,
            record_stats=True,
            task_names=task_names
        )
    elif FLAGS.curriculum_method == "noop":
        print("Using Noop Curriculum")
        curriculum = Constant(0, sample_env.task_space, record_stats=True, task_names=task_names)
    else:
        raise ValueError(f"Unknown curriculum method {FLAGS.curriculum_method}")
    curriculum = make_multiprocessing_curriculum(curriculum, start=False)
    task_space = sample_env.task_space
    del sample_env
    logging.info("curriculum: %s", FLAGS.curriculum_method)
    return curriculum, task_space


omegaconf.OmegaConf.register_new_resolver("uid", uid, use_cache=True)


# Override config_path via --config_path.
@hydra.main(config_path=".", config_name="config", version_base="1.3")
def main(cfg):
    global FLAGS
    FLAGS = cfg

    if not os.path.isabs(FLAGS.savedir):
        FLAGS.savedir = os.path.join(hydra.utils.get_original_cwd(), FLAGS.savedir)

    logging.info("flags:\n%s\n", pprint.pformat(dict(FLAGS)))

    if record.symlink_path(
        FLAGS.savedir, os.path.join(hydra.utils.get_original_cwd(), "latest")
    ):
        logging.info("savedir: %s (symlinked as 'latest')", FLAGS.savedir)
    else:
        logging.info("savedir: %s", FLAGS.savedir)

    train_id = "%s/%s/%s" % (
        FLAGS.entity if FLAGS.entity is not None else getpass.getuser(),
        FLAGS.project,
        FLAGS.group,
    )

    logging.info("train_id: %s", train_id)

    if FLAGS.use_kickstarting:
        student = hackrl.models.create_model(FLAGS, FLAGS.device)
        load_data = torch.load(FLAGS.kickstarting_path)
        t_flags = omegaconf.OmegaConf.create(load_data["flags"])
        teacher = hackrl.models.create_model(t_flags, FLAGS.device)
        teacher.load_state_dict(load_data["learner_state"]["model"])
        model = hackrl.models.KickStarter(
            student, teacher, run_teacher_hs=FLAGS.run_teacher_hs
        )
    else:
        model = hackrl.models.create_model(FLAGS, FLAGS.device)
    optimizer = create_optimizer(model)
    learner_state = LearnerState(model, optimizer)

    model_numel = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Number of model parameters: %i", model_numel)
    record.write_metadata(
        FLAGS.localdir,
        hydra.utils.get_original_cwd(),
        flags=omegaconf.OmegaConf.to_container(FLAGS),
        model_numel=model_numel,
    )

    curriculum = task_space = None
    if FLAGS.syllabus:
        curriculum, task_space = setup_curriculum(FLAGS, model=model)

    envs = moolib.EnvPool(
        lambda: hackrl.environment.create_env(FLAGS, curriculum=curriculum),
        num_processes=FLAGS.num_actor_cpus,
        batch_size=FLAGS.actor_batch_size,
        num_batches=FLAGS.num_actor_batches,
    )

    eval_envs = moolib.EnvPool(
        lambda: hackrl.environment.create_env(FLAGS),
        num_processes=FLAGS.num_actor_cpus,
        batch_size=FLAGS.actor_batch_size,
        num_batches=FLAGS.num_actor_batches,
    )

    if FLAGS.wandb:
        wandb.init(
            project="syllabus-testing",
            config=omegaconf.OmegaConf.to_container(FLAGS),
            group=FLAGS.group,
            entity=FLAGS.entity,
            name=FLAGS.exp_name,
            save_code=True,
        )

    env_states = [EnvBatchState(FLAGS, model) for _ in range(FLAGS.num_actor_batches)]
    eval_env_states = [EnvBatchState(FLAGS, model, actor_batch_size=FLAGS.actor_batch_size)
                       for _ in range(FLAGS.num_actor_batches)]

    rpc = moolib.Rpc()
    rpc.set_name(FLAGS.local_name)
    rpc.connect(FLAGS.connect)

    rpc_group = moolib.Group(rpc, name=train_id)

    accumulator = moolib.Accumulator(
        group=rpc_group,
        name="model",
        parameters=model.parameters(),
        buffers=model.buffers(),
    )
    accumulator.set_virtual_batch_size(FLAGS.virtual_batch_size)

    learn_batcher = moolib.Batcher(FLAGS.batch_size, FLAGS.device, dim=1)

    stats = {
        "mean_episode_return": StatMean(),
        "mean_episode_step": StatMean(),
        "SPS": StatMean(),
        "env_act_steps": StatSum(),
        "env_train_steps": StatSum(),
        "optimizer_steps": StatSum(),
        "running_reward": StatMean(),
        "discounted_running_reward": StatMean(),
        "reward_scale": StatMean(),
        "running_step": StatMean(),
        "steps_done": StatSum(),
        "episodes_done": StatSum(),
        "unclipped_grad_norm": StatMean(),
        "virtual_batch_size": StatMean(),
        "num_gradients": StatMean(),
        "policy_loss": StatMean(),
        "max_policy_lag": StatMean(),
        "mean_policy_lag": StatMean(),
        "min_policy_lag": StatMean(),
        "baseline_loss": StatMean(),
        "max_baseline_value": StatMean(),
        "mean_baseline_value": StatMean(),
        "min_baseline_value": StatMean(),
        "entropy_loss": StatMean(),
        "max_entropy_value": StatMean(),
        "mean_entropy_value": StatMean(),
        "min_entropy_value": StatMean(),
        "clipped_baseline_fraction": StatMean(),
        "clipped_policy_fraction": StatMean(),
        "kickstarting_loss": StatMean(),
        "inverse_loss": StatMean(),
        "inverse_prediction_accuracy": StatMean(),
        "random_inverse_loss": StatMean(),
        "random_inverse_prediction_accuracy": StatMean(),
        "mean_square_discounted_running_reward": StatMean(cumulative=True),
        "reward_normalised": StatMean(),
        "running_advantages": StatMean(cumulative=True),
        "sample_advantages": StatMean(),
        "supervised_loss": StatMean(),
        "mean_final_dungeon_level": StatMean(),
        "mean_final_character_level": StatMean(),
        "mean_final_gold": StatMean(),
        "max_final_dungeon_level": StatMax(),
        "max_final_character_level": StatMax(),
        "max_final_gold": StatMax(),
    }

    stats_allowlist = [
        "mean_episode_return",
        "mean_episode_step",
        "SPS",
        # "running_reward",
        # "discounted_running_reward",
        # "reward_scale",
        # "running_step",
        # "unclipped_grad_norm",
        # "policy_loss",
        # "mean_policy_lag",
        # "baseline_loss",
        "mean_baseline_value",
        # "entropy_loss",
        "mean_entropy_value",
        "reward_normalised",
        "mean_final_dungeon_level",
        "mean_final_character_level",
        "mean_final_gold",
        "max_final_dungeon_level",
        "max_final_character_level",
        "max_final_gold",
    ]
    learner_state.global_stats = copy.deepcopy(stats)
    eval_stats = {k: copy.deepcopy(v) for k, v in stats.items()}

    checkpoint_path = os.path.join(FLAGS.savedir, "checkpoint.tar")

    if os.path.exists(checkpoint_path):
        logging.info("Loading checkpoint: %s" % checkpoint_path)
        load_checkpoint(checkpoint_path, learner_state)
        accumulator.set_model_version(learner_state.model_version)
        logging.info("loaded stats %s", learner_state.global_stats)

    global_stats_accumulator = GlobalStatsAccumulator(
        rpc_group, learner_state.global_stats
    )

    terminate = False
    previous_signal_handler = {}

    def signal_handler(signum, frame):
        nonlocal terminate
        logging.info(
            "Got signal %s, quitting!",
            signal.strsignal(signum) if hasattr(signal, "strsignal") else signum,
        )
        terminate = True
        previous_handler = previous_signal_handler[signum]
        if previous_handler is not None:
            previous_signal_handler[signum] = None
            signal.signal(signum, previous_handler)

    previous_signal_handler[signal.SIGTERM] = signal.signal(
        signal.SIGTERM, signal_handler
    )
    previous_signal_handler[signal.SIGINT] = signal.signal(
        signal.SIGINT, signal_handler
    )

    if torch.backends.cudnn.is_available():
        logging.info("Optimising CuDNN kernels")
        torch.backends.cudnn.benchmark = True

    if FLAGS.supervised_loss or FLAGS.behavioural_clone:
        global TTYREC_ENVPOOL, TTYREC_HIDDEN_STATE
        tp = concurrent.futures.ThreadPoolExecutor(max_workers=FLAGS.ttyrec_cpus)
        TTYREC_HIDDEN_STATE = []
        for _ in range(FLAGS.ttyrec_envpool_size):
            hs = nest.map(
                lambda x: x.to(FLAGS.device),
                model.initial_state(batch_size=FLAGS.ttyrec_batch_size),
            )
            TTYREC_HIDDEN_STATE.append(hs)
        TTYREC_ENVPOOL = make_ttyrec_envpool(tp, FLAGS)

    # Run.
    now = time.time()
    prev_env_train_steps = 0
    prev_global_env_train_steps = 0
    next_env_index = 0
    next_eval_env_index = 0
    last_log = now
    last_reduce_stats = now
    is_leader = False
    is_connected = False
    actor_index = 0
    curriculum.start()
    while not terminate:
        prev_now = now
        now = time.time()

        steps = learner_state.global_stats["env_train_steps"].result()
        if steps >= FLAGS.total_steps:
            logging.info("Stopping training after %i steps", steps)
            break

        rpc_group.update()
        accumulator.update()
        if accumulator.wants_state():
            assert accumulator.is_leader()
            accumulator.set_state(learner_state.save())
        if accumulator.has_new_state():
            assert not accumulator.is_leader()
            learner_state.load(accumulator.state())

        was_connected = is_connected
        is_connected = accumulator.connected()
        if not is_connected:
            if was_connected:
                logging.warning("Training interrupted!")
            # If we're not connected, sleep for a bit so we don't busy-wait
            logging.info("Your training will commence shortly.")
            time.sleep(1)
            continue

        was_leader = is_leader
        is_leader = accumulator.is_leader()
        if not was_connected:
            logging.info(
                "Training started. Leader is %s, %d members, model version is %d"
                % (
                    "me!" if is_leader else accumulator.get_leader(),
                    len(rpc_group.members()),
                    learner_state.model_version,
                )
            )
            prev_global_env_train_steps = learner_state.global_stats[
                "env_train_steps"
            ].result()

        learner_state.train_time += now - prev_now
        if now - last_reduce_stats >= 2:
            last_reduce_stats = now
            global_stats_accumulator.reduce(stats)
        if now - last_log >= FLAGS.log_interval:
            delta = now - last_log
            last_log = now

            global_stats_accumulator.reduce(stats)
            global_stats_accumulator.reset()

            # Evaluate agent on test seeds
            eval_env_states, next_eval_env_index = evaluate_agent(FLAGS, model, eval_envs, eval_env_states, eval_stats,
                                                                  next_eval_env_index=next_eval_env_index)

            prev_env_train_steps = calculate_sps(stats, delta, prev_env_train_steps)
            prev_global_env_train_steps = calculate_sps(
                learner_state.global_stats, delta, prev_global_env_train_steps
            )

            steps = learner_state.global_stats["env_train_steps"].result()

            log(stats, step=steps, is_global=False, allowlist=stats_allowlist)
            log(eval_stats, step=steps, is_global=False, is_eval=True, allowlist=stats_allowlist)
            log(learner_state.global_stats, step=steps, is_global=True, curriculum=curriculum, allowlist=stats_allowlist)

        if is_leader:
            if not was_leader:
                leader_filename = os.path.join(
                    FLAGS.savedir, "leader-%03d" % learner_state.num_previous_leaders
                )
                record.symlink_path(FLAGS.localdir, leader_filename)
                logging.info(
                    "Created symlink %s -> %s", leader_filename, FLAGS.localdir
                )
                learner_state.num_previous_leaders += 1
            if not was_leader and not os.path.exists(checkpoint_path):
                logging.info("Training a new model from scratch.")
            if (
                learner_state.train_time - learner_state.last_checkpoint
                >= FLAGS.checkpoint_interval
            ):
                learner_state.last_checkpoint = learner_state.train_time
                save_checkpoint(checkpoint_path, learner_state)
            if (
                learner_state.train_time - learner_state.last_checkpoint_history
                >= FLAGS.checkpoint_history_interval
            ):
                learner_state.last_checkpoint_history = learner_state.train_time
                save_checkpoint(
                    os.path.join(
                        FLAGS.savedir,
                        "checkpoint_v%d.tar" % learner_state.model_version,
                    ),
                    learner_state,
                )

        if accumulator.has_gradients():
            gradient_stats = accumulator.get_gradient_stats()
            # logging.info("batch_size: %s", gradient_stats["batch_size"])
            # logging.info("virtual_batch_size: %s", stats["virtual_batch_size"])
            # logging.info("num_gradients: %s", gradient_stats["num_gradients"])
            stats["virtual_batch_size"] += gradient_stats["batch_size"]
            stats["num_gradients"] += gradient_stats["num_gradients"]
            step_optimizer(learner_state, stats)
            accumulator.zero_gradients()
        elif not learn_batcher.empty() and accumulator.wants_gradients():
            compute_gradients(learn_batcher.get(), learner_state, stats, curriculum, actor_index)
            actor_index = (actor_index + FLAGS.batch_size) % (FLAGS.actor_batch_size * FLAGS.num_actor_batches)
            accumulator.reduce_gradients(FLAGS.batch_size)
        else:
            if accumulator.wants_gradients():
                accumulator.skip_gradients()

            # Generate data.
            cur_index = next_env_index
            next_env_index = (next_env_index + 1) % FLAGS.num_actor_batches

            env_state = env_states[cur_index]
            if env_state.future is None:
                env_state.future = envs.step(cur_index, env_state.prev_action)
            cpu_env_outputs = env_state.future.result()
            # logging.info("generated data")

            env_outputs = nest.map(
                lambda t: t.to(FLAGS.device, copy=True), cpu_env_outputs
            )

            env_outputs["prev_action"] = env_state.prev_action
            prev_core_state = env_state.core_state
            model.eval()
            with torch.no_grad():
                actor_outputs, env_state.core_state = model(
                    nest.map(lambda t: t.unsqueeze(0), env_outputs),
                    env_state.core_state,
                )
            # logging.info("generated actions")
            actor_outputs = nest.map(lambda t: t.squeeze(0), actor_outputs)
            action = actor_outputs["action"]
            env_state.update(cpu_env_outputs, action, stats)
            if curriculum is not None and FLAGS.curriculum_method == "centralplr":
                next_value = actor_outputs["baseline"]
                tasks = [task_space.decode(int(t)) for t in cpu_env_outputs["tty_cursor"]]
                rewards = cpu_env_outputs["reward"]
                if FLAGS.rms_reward_norm and FLAGS.per_task_reward_norm and curriculum is not None:
                    reward_tasks = cpu_env_outputs["tty_cursor"]
                    stds = {t: curriculum.curriculum.stat_recorder.episode_returns[t].std(
                    ) for t in range(curriculum.num_tasks)}

                    # Check for invalid std dev values
                    for k, v in stds.items():
                        if v is None or math.isnan(v):
                            stds[k] = stats["mean_square_discounted_running_reward"].mean() ** 0.5

                    reward_std = reward_tasks.cpu().apply_(lambda t: stds[int(t)])

                    reward_scale = torch.clamp(reward_std, min=0.01)
                    rewards /= reward_scale.cpu()
                update = {
                    "update_type": "on_demand",
                    "metrics": {
                        "value": actor_outputs["baseline"].view(-1, 1),
                        "next_value": next_value.view(-1, 1),
                        "rew": rewards,
                        "dones": cpu_env_outputs["done"],
                        "tasks": tasks,
                    },
                }
                curriculum.update(update)

            del cpu_env_outputs  # envs.step invalidates cpu_env_outputs.
            env_state.future = envs.step(cur_index, action)

            stats["env_act_steps"] += action.numel()

            last_data = {
                "env_outputs": env_outputs,
                "actor_outputs": actor_outputs,
            }
            env_state.time_batcher.stack(last_data)

            if not env_state.time_batcher.empty():
                data = env_state.time_batcher.get()
                data["initial_core_state"] = env_state.initial_core_state
                learn_batcher.cat(data)

                # We need the last entry of the previous time batch
                # to be put into the first entry of this time batch,
                # with the initial_core_state to match
                env_state.initial_core_state = prev_core_state
                env_state.time_batcher.stack(last_data)
    if is_connected and is_leader:
        save_checkpoint(checkpoint_path, learner_state)
    tp.shutdown()
    logging.info("Graceful exit. Bye bye!")


if __name__ == "__main__":
    main()
