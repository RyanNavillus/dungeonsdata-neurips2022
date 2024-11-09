#!/bin/bash

# The following are the commands used to run the experiment code on SLURM
# In particular note that 
#   1) --broker arguments are the IP & Port for a moolib broker
#   2) --constraint and --cpu arguments reflect the machine constraints used
#   3) --time is set for 3 days, although in practice many of these finish well before.
#   4) --exp_set and --exp_point are just arguments to label the experimental runs

# APPO Experiments
#python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="Challenge" run_id=1 syllabus=False
#python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="Challenge" run_id=2 syllabus=False
#python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="Challenge" run_id=3 syllabus=False
#python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="Challenge" run_id=4 syllabus=False
#python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="Challenge" run_id=5 syllabus=False

# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="Challenge" run_id=6 syllabus=False
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="Challenge" run_id=7 syllabus=False
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="Challenge" run_id=8 syllabus=False
#python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="Challenge" run_id=9 syllabus=False
#python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="Challenge" run_id=10 syllabus=False

# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="Sequential" curriculum_method="sq" run_id=1
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="Sequential" curriculum_method="sq" run_id=2
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="Sequential" curriculum_method="sq" run_id=3
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="Sequential" curriculum_method="sq" run_id=4
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="Sequential" curriculum_method="sq" run_id=5


# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="Domain_Randomization" curriculum_method="dr" run_id=6
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="Domain_Randomization" curriculum_method="dr" run_id=7
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="Domain_Randomization" curriculum_method="dr" run_id=8
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="Domain_Randomization" curriculum_method="dr" run_id=9
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="Domain_Randomization" curriculum_method="dr" run_id=10

# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="Sequential" curriculum_method="sq" run_id=6
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="Sequential" curriculum_method="sq" run_id=7
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="Sequential" curriculum_method="sq" run_id=8
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="Sequential" curriculum_method="sq" run_id=9
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="Sequential" curriculum_method="sq" run_id=10:

# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="DR_Seed" curriculum_method="dr" run_id=11
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="DR_Seed" curriculum_method="dr" run_id=12
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="DR_Seed" curriculum_method="dr" run_id=13
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="DR_Seed" curriculum_method="dr" run_id=14
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="DR_Seed" curriculum_method="dr" run_id=15


# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="DR_Seed_4000" curriculum_method="dr" run_id=16 num_seeds=4000
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="DR_Seed_4000" curriculum_method="dr" run_id=17 num_seeds=4000
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="DR_Seed_4000" curriculum_method="dr" run_id=18 num_seeds=4000
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="DR_Seed_4000" curriculum_method="dr" run_id=19 num_seeds=4000
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="DR_Seed_4000" curriculum_method="dr" run_id=20 num_seeds=4000

# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="DR_Seed_80000" curriculum_method="dr" run_id=21 num_seeds=80000
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="DR_Seed_80000" curriculum_method="dr" run_id=22 num_seeds=80000
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="DR_Seed_80000" curriculum_method="dr" run_id=23 num_seeds=80000
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="DR_Seed_80000" curriculum_method="dr" run_id=24 num_seeds=80000
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="DR_Seed_80000" curriculum_method="dr" run_id=25 num_seeds=80000

# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="PLR_Seed_200" curriculum_method="centralplr" run_id=26 num_seeds=200
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="PLR_Seed_200" curriculum_method="centralplr" run_id=27 num_seeds=200
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="PLR_Seed_200" curriculum_method="centralplr" run_id=28 num_seeds=200
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="PLR_Seed_200" curriculum_method="centralplr" run_id=29 num_seeds=200
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="PLR_Seed_200" curriculum_method="centralplr" run_id=30 num_seeds=200

# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="PLR_Seed_4000" curriculum_method="centralplr" run_id=31 num_seeds=4000
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="PLR_Seed_4000" curriculum_method="centralplr" run_id=32 num_seeds=4000
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="PLR_Seed_4000" curriculum_method="centralplr" run_id=33 num_seeds=4000
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="PLR_Seed_4000" curriculum_method="centralplr" run_id=34 num_seeds=4000
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="PLR_Seed_4000" curriculum_method="centralplr" run_id=35 num_seeds=4000

# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=16 exp_set=2G num_actor_cpus=8 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="Simple_PLR_Seed_2_200" curriculum_method="simpleplr" run_id=46 num_seeds=200
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=16 exp_set=2G num_actor_cpus=8 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="Simple_PLR_Seed_2_200" curriculum_method="simpleplr" run_id=47 num_seeds=200
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=16 exp_set=2G num_actor_cpus=8 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="Simple_PLR_Seed_2_200" curriculum_method="simpleplr" run_id=48 num_seeds=200
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=16 exp_set=2G num_actor_cpus=8 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="Simple_PLR_Seed_2_200" curriculum_method="simpleplr" run_id=49 num_seeds=200
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=16 exp_set=2G num_actor_cpus=8 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="Simple_PLR_Seed_2_200" curriculum_method="simpleplr" run_id=50 num_seeds=200

# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=16 exp_set=2G num_actor_cpus=8 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="Simple_PLR_Seed_2_4000" curriculum_method="simpleplr" run_id=56 num_seeds=4000
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=16 exp_set=2G num_actor_cpus=8 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="Simple_PLR_Seed_2_4000" curriculum_method="simpleplr" run_id=57 num_seeds=4000
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=16 exp_set=2G num_actor_cpus=8 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="Simple_PLR_Seed_2_4000" curriculum_method="simpleplr" run_id=58 num_seeds=4000
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=16 exp_set=2G num_actor_cpus=8 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="Simple_PLR_Seed_2_4000" curriculum_method="simpleplr" run_id=59 num_seeds=4000
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=16 exp_set=2G num_actor_cpus=8 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="Simple_PLR_Seed_2_4000" curriculum_method="simpleplr" run_id=60 num_seeds=4000

# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=16 exp_set=2G num_actor_cpus=8 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="DR_Seed_2_4000" curriculum_method="dr" run_id=61 num_seeds=4000
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=16 exp_set=2G num_actor_cpus=8 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="DR_Seed_2_4000" curriculum_method="dr" run_id=62 num_seeds=4000
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=16 exp_set=2G num_actor_cpus=8 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="DR_Seed_2_4000" curriculum_method="dr" run_id=63 num_seeds=4000
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=16 exp_set=2G num_actor_cpus=8 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="DR_Seed_2_4000" curriculum_method="dr" run_id=64 num_seeds=4000
# python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=16 exp_set=2G num_actor_cpus=8 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="DR_Seed_2_4000" curriculum_method="dr" run_id=65 num_seeds=4000

#python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --constraint=volta32gb --cpus=20 exp_set=2G num_actor_cpus=20 exp_point=monk-APPO  total_steps=2_000_000_000 character='mon-hum-neu-mal'

python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=16 exp_set=2G num_actor_cpus=8 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="DR_debug" curriculum_method="dr" run_id=66 num_seeds=4000
python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --cpus=16 exp_set=2G num_actor_cpus=8 exp_point=@-APPO     total_steps=2_000_000_000 exp_name="PLR_debug" curriculum_method="simpleplr" run_id=67 num_seeds=4000
# Behavioural Cloning Experiments
#python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --constraint=volta32gb --cpus=20 exp_set=2G  exp_point=@-AA-BC     num_actor_cpus=20 total_steps=2_000_000_000 actor_batch_size=256 batch_size=128 ttyrec_batch_size=512 supervised_loss=1 adam_learning_rate=0.001 behavioural_clone=True 
#python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --constraint=volta32gb --cpus=20 exp_set=2G  exp_point=monk-AA-BC   num_actor_cpus=20 total_steps=2_000_000_000 actor_batch_size=256 batch_size=128 ttyrec_batch_size=512 supervised_loss=1 adam_learning_rate=0.001 behavioural_clone=True character='mon-hum-neu-mal'
#python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --constraint=volta32gb --cpus=20 exp_set=2G  exp_point=@-NAO-BC num_actor_cpus=20 total_steps=2_000_000_000 actor_batch_size=256 batch_size=128 ttyrec_batch_size=512 supervised_loss=1 adam_learning_rate=0.001 behavioural_clone=True dataset=nld-nao dataset_bootstrap_actions=True bootstrap_pred_max=True  dataset_bootstrap_path=/path/to/checkpoint.tar

# APPO + Behavioural Cloning Experiments
#python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --constraint=volta32gb --cpus=20 total_steps=2_000_000_000 num_actor_cpus=20 ttyrec_batch_size=256 supervised_loss=0.1 exp_set=2G exp_point=@-APPO-AA-BC
#python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --constraint=volta32gb --cpus=20 total_steps=2_000_000_000 num_actor_cpus=20 ttyrec_batch_size=256 supervised_loss=0.1 exp_set=2G exp_point=monk-APPO-AA-BC character='mon-hum-neu-mal'  
#python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --constraint=volta32gb --cpus=20 total_steps=2_000_000_000 num_actor_cpus=20 ttyrec_batch_size=256 supervised_loss=0.1 exp_set=2G exp_point=@-APPO-NAO-crudeBC-all dataset=nld-nao dataset_bootstrap_actions=True bootstrap_pred_max=True num_actor_cpus=20 dataset_bootstrap_path=/path/to/checkpoint.tar

# APPO +  Kickstarting Experiments
#python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --constraint=volta32gb --cpus=20 exp_set=2G  exp_point=@-APPO-AA-KS      total_steps=2_000_000_000 num_actor_cpus=20 kickstarting_loss=0.1 use_kickstarting=true kickstarting_path=/checkpoint/ehambro/20220531/fierce-snail/checkpoint.tar
#python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --constraint=volta32gb --cpus=20 exp_set=2G  exp_point=monk-APPO-AA-KS   total_steps=2_000_000_000 num_actor_cpus=20 kickstarting_loss=0.1 use_kickstarting=true kickstarting_path=/checkpoint/ehambro/20220531/blazing-slug/checkpoint.tar  character='mon-hum-neu-mal'  
#python scripts/sbatch_experiment.py --broker $BROKER_IP:$BROKER_PORT --time=4320 --constraint=volta32gb --cpus=20 exp_set=2G  exp_point=@-APPO-NAO-KS-all total_steps=2_000_000_000 num_actor_cpus=20 kickstarting_loss=0.1 use_kickstarting=true kickstarting_path=/checkpoint/ehambro/20220531/celadon-llama/checkpoint.tar  dataset=nld-nao dataset_bootstrap_actions=True bootstrap_pred_max=True

