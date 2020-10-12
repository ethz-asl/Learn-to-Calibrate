#!/usr/bin/env python


import numpy as np
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.agents.ddpg import DDPGTrainer
from ray.tune import run, sample_from
import random
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf

tf = try_import_tf()
from calibration_env import CamCalibrEnv, CamCalibrEnv_seq
import ray

#model defination (hidden state output)
class MyKerasModel(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(MyKerasModel, self).__init__(obs_space, action_space,
                                           num_outputs, model_config, name)



        #input
        self.input = tf.keras.layers.Input(shape=(None,40))

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
    "env": CamCalibrEnv,
    "gamma": 0.99,
    "num_workers": 1,
    "num_envs_per_worker": 1,
    "timesteps_per_iteration": 6,
    # twin Q-net
    "twin_q": True,
    # delayed policy update
    "policy_delay": 2,
    # target policy smoothing
    # (this also replaces OU exploration noise with IID Gaussian exploration
    # noise, for now)
    "smooth_target_policy": True,
    # gaussian stddev of target action noise for smoothing
    "target_noise": 0.02,
    # target noise limit (bound)
    "target_noise_clip": 0.03,
    "train_batch_size": 16,
    "learning_starts": 30,
    "horizon": 6,
    "actor_hiddens": [300],
    # Hidden layers activation of the postprocessing stage of the policy
    # network
    "actor_hidden_activation": "relu",
    # Postprocess the critic network model output with these hidden layers;
    # again, if use_state_preprocessor is True, then the state will be
    # preprocessed by the model specified with the "model" config option first.
    "critic_hiddens": [300],
    # Hidden layers activation of the postprocessing state of the critic.
    "critic_hidden_activation": "relu",
    "lr": 1e-4, #1e-4
    "clip_actions": False,
    "exploration_config": {
        # DDPG uses OrnsteinUhlenbeck (stateful) noise to be added to NN-output
        # actions (after a possible pure random phase of n timesteps).
        "type": "OrnsteinUhlenbeckNoise",
        # For how many timesteps should we return completely random actions,
        # before we start adding (scaled) noise?
        "random_timesteps": 30,
        # The OU-base scaling factor to always apply to action-added noise.
        "ou_base_scale": 0.1,
        # The OU theta param.
        "ou_theta": 0.15,
        # The OU sigma param.
        "ou_sigma": 0.2,
        # The initial noise scaling factor.
        "initial_scale": 1.0,
        # The final noise scaling factor.
        "final_scale": 0.02,
        # Timesteps over which to anneal scale (from initial to final values).
        "scale_timesteps": 100
    },

    "model":{
        "custom_model": "my_model",
    },
    "evaluation_interval": 5,
    "evaluation_num_episodes": 1,
}

#trainable function
def train(config, reporter):
    trainer = DDPGTrainer(config=config, env=CamCalibrEnv_seq)
    while True:
        result = trainer.train()
        reporter(**result)
        if result["timesteps_since_restore"] > 250:
            phase = 1
        else:
            phase = 0
        trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_phase(phase)))

#train
analysis=tune.run(train,
                  stop={"timesteps_total": 400},
                  name="cal_ddpg_test",
                  config = config,
                  )
