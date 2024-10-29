from nle.env import tasks
from syllabus.examples.task_wrappers import NetHackCollect, NetHackDescend, NetHackScoutClipped, NetHackSatiate, NetHackSeed

ENVS = dict(
    staircase=tasks.NetHackStaircase,
    score=tasks.NetHackScore,
    pet=tasks.NetHackStaircasePet,
    oracle=tasks.NetHackOracle,
    gold=tasks.NetHackGold,
    eat=tasks.NetHackEat,
    scout=tasks.NetHackScout,
    challenge=tasks.NetHackChallenge,
    collect=NetHackCollect,
    descend=NetHackDescend,
    scout_clipped=NetHackScoutClipped,
    satiate=NetHackSatiate,
    seed=NetHackSeed,
)
