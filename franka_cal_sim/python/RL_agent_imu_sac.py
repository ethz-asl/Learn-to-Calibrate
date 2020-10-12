#!/usr/bin/env python


import numpy as np
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.agents.sac import SACTrainer
from ray.tune import run, sample_from
import random
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf

tf = try_import_tf()
from calibration_env import CamCalibrEnv, CamCalibrEnv_seq, imuCalibrEnv_seq
import ray

import psutil
import gc

#model defination (hidden state output)
class MyKerasModel(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(MyKerasModel, self).__init__(obs_space, action_space,
                                           num_outputs, model_config, name)



        #input
        self.input = tf.keras.layers.Input(shape=(None,48))

        #network (recurrent model out)
        mask_state_input = tf.keras.layers.Masking(mask_value=0.)(self.input)
        #obs_h1 = tf.keras.layers.Dense(256, activation='relu')(mask_state_input)
        actor_rnn,state_h = tf.keras.layers.GRU(256, return_state=True)(mask_state_input)
        self.output = state_h

        self.base_model = tf.keras.Model(self.input, self.output)
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.base_model(input_dict["obs"])
        return model_out, state


#init
ray.init()

#model registration
ModelCatalog.register_custom_model("my_model", MyKerasModel)

#config
config={
    "env": imuCalibrEnv_seq,
    "gamma": 0.99,
    "num_workers": 1,
    "num_envs_per_worker": 1,
    "timesteps_per_iteration": 1,
    # twin Q-net
    "twin_q": True,
    "train_batch_size": 64,
    "learning_starts": 8,
    "horizon": 3,

    # === Model ===
    "use_state_preprocessor": True,
    # RLlib model options for the Q function(s).
    "Q_model": {
        "fcnet_activation": "relu",
        "fcnet_hiddens": [256],
    },
    # RLlib model options for the policy function.
    "policy_model": {
        "fcnet_activation": "relu",
        "fcnet_hiddens": [256],
    },
    # Unsquash actions to the upper and lower bounds of env's action space.
    # Ignored for discrete action spaces.
    "clip_actions": False,
    "normalize_actions": False,

    # === Learning ===
    # Disable setting done=True at end of episode. This should be set to True
    # for infinite-horizon MDPs (e.g., many continuous control problems).
    "no_done_at_end": False,
    # Update the target by \tau * policy + (1-\tau) * target_policy.
    "tau": 1e-2,
    # Initial value to use for the entropy weight alpha.
    "initial_alpha": 1.0,
    # Target entropy lower bound. If "auto", will be set to -|A| (e.g. -2.0 for
    # Discrete(2), -3.0 for Box(shape=(3,))).
    # This is the inverse of reward scale, and will be optimized automatically.
    "target_entropy": "auto",
    # N-step target updates.
    "n_step": 1,


    # === Replay buffer ===
    # Size of the replay buffer. Note that if async_updates is set, then
    # each worker will have a replay buffer of this size.
    "buffer_size": int(5000),
    # If True prioritized replay buffer will be used.
    "prioritized_replay": True,
    "prioritized_replay_alpha": 0.6,
    "prioritized_replay_beta": 0.4,
    "prioritized_replay_eps": 1e-6,
    "prioritized_replay_beta_annealing_timesteps": 1000,
    "final_prioritized_replay_beta": 0.4,

    "compress_observations": False,

    # === Optimization ===
    "optimization": {
        "actor_learning_rate": 1e-5,
        "critic_learning_rate": 1e-4,
        "entropy_learning_rate": 1e-3,
    },

    "model":{
        "custom_model": "my_model",
    },
    }

def auto_garbage_collect(pct=80.0):
    """
    auto_garbage_collection - Call the garbage collection if memory used is greater than 80% of total available memory.
                              This is called to deal with an issue in Ray not freeing up used memory.

        pct - Default value of 80%.  Amount of memory in use that triggers the garbage collection call.
    """
    if psutil.virtual_memory().percent >= pct:
        gc.collect()
    return

#trainable function
def train(config, reporter):
    trainer = SACTrainer(config=config, env=imuCalibrEnv_seq)
    #checkpoint_path = trainer.save()
    policy = trainer.get_policy()
    print(policy.dist_class)

    i = 0
    while True:
        result = trainer.train()
        reporter(**result)
        # if result["timesteps_since_restore"] > 200:
        #     phase = 1
        # else:
        #     phase = 0
        # trainer.workers.foreach_worker(
        #     lambda ev: ev.foreach_env(
        #         lambda env: env.set_phase(phase)))
        # if i==0:
        #     trainer.restore("/home/yunke/ray_results/SAC_imuCalibrEnv_seq_2020-05-21_23-27-20ig3rw_2c/checkpoint_1/checkpoint-1")
        if i%100==0:
            checkpoint_path = trainer.save()
            print(checkpoint_path)
        auto_garbage_collect()
        i+=1
#train
analysis=tune.run(train,
                  stop={"timesteps_total": 20000},
                  name="cal_sac_test_new",
                  config = config
                  )
