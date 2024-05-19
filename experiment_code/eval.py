import os
import sys

import moolib
import numpy as np
import omegaconf
import torch
import tqdm
import wandb

import hackrl.core
import hackrl.environment
import hackrl.models
from hackrl.core import nest

ENVS = None


def load_model_and_flags(path, device, character):
    load_data = torch.load(path, map_location=torch.device(device))
    flags = omegaconf.OmegaConf.create(load_data["flags"])
    flags.device = device
    flags.character = character
    flags.syllabus = True
    model = hackrl.models.create_model(flags, device)
    if flags.use_kickstarting:
        print("Kickstarting")
        t_data = torch.load(flags.kickstarting_path)
        t_flags = omegaconf.OmegaConf.create(t_data["flags"])
        teacher = hackrl.models.create_model(t_flags, flags.device)
        # teacher.load_state_dict(load_data["learner_state"]["model"])
        model = hackrl.models.KickStarter(
            model, teacher, run_teacher_hs=flags.run_teacher_hs
        )
    model.load_state_dict(load_data["learner_state"]["model"])
    return model, flags


def generate_envpool_rollouts(model, rollouts, flags, pbar_idx=0, SCORE=False):
    global ENVS
    # NB: We do NOT want to generate the first N rollouts from B batch
    # of envs since this will bias short episodes.
    # Instead lets just allocate some episodes to each env
    num_batches = 2
    split = 4
    rollouts = rollouts // (num_batches * split)
    flags.batch_size = rollouts
    if SCORE:
        flags.env = {'name': 'score', 'max_episode_steps': 100000}
    device = flags.device

    ENVS = moolib.EnvPool(
        lambda: hackrl.environment.create_env(flags),
        num_processes=40,
        batch_size=rollouts,
        num_batches=num_batches,
    )

    rollouts_left = (
        torch.ones(
            (
                num_batches,
                rollouts,
            )
        )
        .long()
        .to(device)
        * split
    )
    current_reward = torch.zeros(
        (
            num_batches,
            rollouts,
        )
    ).to(device)

    returns = []
    results = [None, None]
    grand_pbar = tqdm.tqdm(position=0, leave=True)
    pbar = tqdm.tqdm(
        total=rollouts * num_batches * split, position=pbar_idx + 1, leave=True
    )

    action = torch.zeros((num_batches, rollouts)).long().to(device)
    hs = [model.initial_state(rollouts) for _ in range(num_batches)]
    hs = nest.map(lambda x: x.to(device), hs)

    totals = torch.sum(rollouts_left).item()
    subtotals = [torch.sum(rollouts_left[i]).item() for i in range(num_batches)]
    while totals > 0:
        grand_pbar.update(1)
        for i in range(num_batches):
            if subtotals[i] == 0:
                continue
            if results[i] is None:
                results[i] = ENVS.step(i, action[i])
            outputs = results[i].result()

            env_outputs = nest.map(lambda t: t.to(flags.device, copy=True), outputs)
            env_outputs["prev_action"] = action[i]
            current_reward += env_outputs["reward"]

            done_and_valid = env_outputs["done"].int() * rollouts_left[i].bool().int()
            finished = torch.sum(done_and_valid).item()
            totals -= finished
            subtotals[i] -= finished

            for j in np.argwhere(done_and_valid.cpu().numpy()):
                reward = current_reward[i][j[0]].item()
                returns.append(reward)
                #wandb.log({"episode_return": reward})

            current_reward[i] *= 1 - env_outputs["done"].int()
            rollouts_left[i] -= done_and_valid
            if finished:
                pbar.update(finished)

            env_outputs = nest.map(lambda x: x.unsqueeze(0), env_outputs)
            with torch.no_grad():
                outputs, hs[i] = model(env_outputs, hs[i])
                
            action[i] = outputs["action"].reshape(-1)
            results[i] = ENVS.step(i, action[i])
    print("Character:", flags.character)
    returns.sort(reverse=True)
    print("Top 5:", returns[:5])
    print(np.mean(returns[:5]))
    return len(returns), np.mean(returns), np.std(returns), np.median(returns), (returns, returns[:5], np.mean(returns[:5]), flags.character)


def find_checkpoint(path, min_steps, device):

    return f"{path}/checkpoint.tar" # select a checkpoint to evaluate

    versions = []
    flags = omegaconf.OmegaConf.create(
        torch.load(path + "/checkpoint.tar", map_location=torch.device(device))["flags"]
    )
    for f in os.listdir(path):
        ff = str(f)
        if ff.startswith("checkpoint_v"):
            v = int(ff.replace("checkpoint_v", "").replace(".tar", ""))
            versions.append(v)
    desired_v = min_steps / (flags.batch_size * flags.unroll_length)
    for v in sorted(versions):
        allowed_paths = [
            "/checkpoint/ehambro/20220531/meek-binturong",
            "/checkpoint/ehambro/20220531/adamant-viper",
            "/checkpoint/ehambro/20220531/hallowed-bat",
            "/checkpoint/ehambro/20220531/celadon-llama",
        ]
        if v > desired_v or v > 122070.325 or (path in allowed_paths and v > 18000):
            return f"{path}/checkpoint_v{v}.tar"
    return f"{path}/checkpoint_v{v}.tar"

    print("Returning checkpoint.tar")
    return f"{path}/checkpoint.tar"


def evaluate_folder(name, path, min_steps, device, rollouts, character, pbar_idx, SCORE):
    p_ckpt = find_checkpoint(path, min_steps, device)
    if not p_ckpt:
        print(f"Not yet: {name} - {path}")
        return (
            name,
            path,
            -1,
            -1,
            -1,
            -1,
        )
    print(f"{pbar_idx} {name} Using: {p_ckpt}")
    os.makedirs(f"{DIR}/{NAME}/", exist_ok=True)
    model, flags = load_model_and_flags(p_ckpt, device, character)
    returns = generate_envpool_rollouts(model, rollouts, flags, pbar_idx, SCORE)
    return (name, p_ckpt) + returns


if __name__ == "__main__":
    #DIR = "eval_results/@_txt"
    #DIR = "eval_results/mon-hum-neu-mal_txt"
    #DIR = "eval_results/tou-hum-neu-fem_txt"
    #DIR = "eval_results/val-dwa-law-fem_txt"
    #DIR = "eval_results/wiz-elf-cha-mal_txt"
    
    #DIR = "eval_results/SQ/SQ_1/mon-hum-neu-mal_txt"
    
    NAME, PATH = sys.argv[1], sys.argv[2]
    SCORE = False if (len(sys.argv) == 3 or sys.argv[3] != 'score') else True
    #DEVICE = sys.argv[3] if len(sys.argv) == 4 else "cuda:0"
    DEVICE = sys.argv[4] if len(sys.argv) == 5 else "cuda:0"
    print(f"Running {NAME} - {PATH} on {DEVICE}")
    MIN_STEPS = 1_000_000_000
    ROLLOUTS = 1000 # Original: 1024

    DIR = f"eval_results/{PATH.split('/')[-1].split('_')[0]}/{PATH.split('/')[-1]}"

    #character = "@"
    #character = "mon-hum-neu-mal"
    #character = "tou-hum-neu-fem"
    #character = "val-dwa-law-fem"
    #character = "wiz-elf-cha-mal"
    character = NAME

    # wandb.init(
    #     project='nethack',
    #     group='group2',
    #     entity=None,
    # )

    results = (NAME, PATH, -1, -1, -1)
    results = evaluate_folder(NAME, PATH, MIN_STEPS, DEVICE, ROLLOUTS, character, 0, SCORE)
    print(
        f"{results[0]} Done {results[1]}  Mean {results[3]} ± {results[4]}  | Median {results[5]}"
    )
    if results[2] > -2:
        data = (
            MIN_STEPS,
            ROLLOUTS,
        ) + results
        os.makedirs(f"{DIR}/{NAME}/", exist_ok=True)
        if not SCORE:
            file_name = f"{PATH.split('/')[-1]}_NetHackChallenge.txt"
        else:
            file_name = f"{PATH.split('/')[-1]}_NetHackScore.txt"
        with open(f"{DIR}/{NAME}/{file_name}", "w") as f:
            f.write(",".join(str(d) for d in data) + "\n")
    print("done")
