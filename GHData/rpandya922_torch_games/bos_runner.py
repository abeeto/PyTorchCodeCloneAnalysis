import gym
import torch as th
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.logger import configure

from lta_ppo import LTA_PPO
from lta_features import LTAExtractor
from BoSEnv import RepeatedBoSEnv
from utils import helpful_partner, adversarial_partner, bach_partner, stravinsky_partner
from autoencoder import Autoencoder

def eval_model(model, env, n_eval=50, adapt=False, trained_steps=0, log=False):
    """
    Evaluates a pretrained policy 

    adapt: whether to adapt the model online during evaluation
    trained_steps: number of steps the model was trained for previously
        used for logging, only applicable if adapt=True
    """

    obs = env.reset()
    n_correct = 0
    corrects = np.zeros(n_eval)
    rewards = []

    # keep track of observations and human actions during evaluation
    observations = []
    human_actions = []
    h_pred_entropy_losses = []
    h_pred_acc = []
    ce_loss = th.nn.CrossEntropyLoss()
    if adapt:
        # unfreeze human prediction model, freeze everything else
        for param in model.policy.parameters():
            param.requires_grad = False
        for (name, param) in model.policy.features_extractor.named_parameters():
            if name[:5] == "human":
                param.requires_grad = True

    for i in range(n_eval):
        obs_tensor = obs_as_tensor(np.array([obs]), model.device)
        actions, values, log_probs = model.policy(obs_tensor)

        preprocessed_obs = preprocess_obs(obs_tensor, model.observation_space)

        if model.policy.features_extractor.strategy_encoder is not None:
            latent_state = model.policy.features_extractor.strategy_encoder.encoder(preprocessed_obs[:,4:])
            preprocessed_obs = th.cat((preprocessed_obs, latent_state), 1)

        human_pred = model.policy.features_extractor.human(preprocessed_obs)

        next_obs, rew, done, _ = env.step(actions[0])

        next_h_action = next_obs[1]
        pred_action = th.argmax(human_pred)

        # store data for adaptation
        # don't store if this is the first step after the env has been reset
        if env.game_num != 1:
            observations.append(preprocessed_obs)
            human_actions.append(next_h_action)

        if next_h_action == pred_action:
            corrects[i] = 1
            n_correct += 1

        rewards.append(rew)
        obs = next_obs
        if done:
            obs = env.reset()

        # run adaptation
        if adapt and i > 1:
            n_grad_steps = 3
            for j in range(n_grad_steps):
                obs_tensor = th.concat(observations, 0)
                pred_actions = model.policy.features_extractor.human(obs_tensor)
                next_actions = F.one_hot(th.tensor(human_actions), num_classes=2).float()

                pred_action_labels = th.argmax(pred_actions, dim=1)
                next_action_labels = th.argmax(next_actions, dim=1)
                n_correct_pred = th.sum(pred_action_labels == next_action_labels).item()

                # compute cross entropy loss
                loss = ce_loss(pred_actions, next_actions)

                # Optimization step
                model.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(model.policy.parameters(), model.max_grad_norm)
                model.policy.optimizer.step()

            # saving data for logging
            h_pred_entropy_losses.append(loss.item())
            h_pred_acc.append(n_correct_pred / pred_actions.shape[0])

            if log:
                model.logger.record("adapt/reward", rew)
                model.logger.record("adapt/h_pred_loss", loss.item())
                model.logger.record("adapt/h_pred_acc", (n_correct_pred / pred_actions.shape[0]))
                model.logger.dump(step=trained_steps+i)
        elif not adapt:
            if log:
                model.logger.record("adapt/reward_no_adapt", rew)
                model.logger.dump(step=trained_steps+i)

    rolling_avg = np.convolve(corrects, np.ones(20)/20, mode='valid')

    return n_correct / n_eval, np.mean(rewards), rewards

if __name__ == "__main__":
    th.manual_seed(0)
    np.random.seed(0)

    horizon = 20
    train_timesteps = 60_000
    n_eval = 100
    training_partners = [stravinsky_partner, helpful_partner]
    testing_partner = [adversarial_partner]
    log_testing = True
    use_encoder = False
    train_vanilla = False

    if use_encoder:
        # load autoencoder
        ae_model = Autoencoder(n_input=6, n_latent=1, n_output=2)
        ae_model.load_state_dict(th.load("./data/autoencoder.pt"))
    else:
        ae_model = None

    env = RepeatedBoSEnv(partner_policies=training_partners, horizon=horizon)
    model = LTA_PPO(
        policy=ActorCriticPolicy, 
        env=env,
        policy_kwargs={"features_extractor_class": LTAExtractor,
                       "features_extractor_kwargs": {"features_dim": 34,#16, 
                                                     "n_actions": 2,
                                                     "human_pred": True,
                                                     "strategy_encoder": ae_model}},
        tensorboard_log="./lta_ppo_bos_tensorboard/"
    )

    # TODO: consider adding joint reasoning back so number of features stays the same
    model_ppo = LTA_PPO(
        policy=ActorCriticPolicy, 
        env=env,
        policy_kwargs={"features_extractor_class": LTAExtractor,
                       "features_extractor_kwargs": {"features_dim": 32, 
                                                     "n_actions": 2,
                                                     "human_pred": False,
                                                     "strategy_encoder": None}},
        # tensorboard_log="./lta_ppo_bos_tensorboard/"
    )

    acc, avg_rew, _ = eval_model(model, env, n_eval=n_eval, log=False)
    print("----------------------------------------------------------")
    print("Before training")
    print(f"Average Reward: {avg_rew}  Human Model Accuracy: {acc*100}%")
    print("----------------------------------------------------------")
    print()

    model.learn(total_timesteps=train_timesteps)

    if train_vanilla:
        model_ppo.learn(total_timesteps=train_timesteps)

    # save learned weights to tmp file
    model.save("./data/tmp_lta_ppo")

    # testing effect of human pred network on joint reasoning output 
    # joint_in = th.zeros(34).float()
    # print(th.autograd.functional.jacobian(model.policy.features_extractor.joint, joint_in)[:,:2])

    # measure accuracy of human prediction model
    acc, avg_rew, _ = eval_model(model, env, n_eval=n_eval, log=False)
    print("----------------------------------------------------------")
    print("After training")
    print(f"Average Reward: {avg_rew}  Human Model Accuracy: {acc*100}%")
    print("----------------------------------------------------------")
    print()

    # test on different human partner policy than training
    env2 = RepeatedBoSEnv(partner_policies=testing_partner, horizon=horizon)
    # measure accuracy of human prediction model
    acc, avg_rew, test_rewards = eval_model(model, env2, n_eval=n_eval, adapt=False, 
        trained_steps=train_timesteps, log=log_testing)
    print("----------------------------------------------------------")
    print("After training (different human partner)")
    print(f"Average Reward: {avg_rew}  Human Model Accuracy: {acc*100}%")
    print("----------------------------------------------------------")
    print()

    # test on different human partner policy than training
    env2 = RepeatedBoSEnv(partner_policies=testing_partner, horizon=horizon)
    # measure accuracy of human prediction model
    acc, adapt_avg_rew, _ = eval_model(model, env2, n_eval=n_eval, adapt=True, 
        trained_steps=train_timesteps, log=log_testing)
    print("----------------------------------------------------------")
    print("[with adaptation] After training (different human partner)")
    print(f"Average Reward: {adapt_avg_rew}  Human Model Accuracy: {acc*100}%")
    print("----------------------------------------------------------")

    # log hyperparamaters for this run
    model.logger.output_formats[0].writer.add_hparams({"training_partners": str([f.__name__ for f in training_partners]),
                                                       "testing_partner": testing_partner[0].__name__,
                                                       "horizon": horizon,
                                                       "n_eval": n_eval,
                                                       "train_timesteps": train_timesteps}, 
                                                      {"adapt/adapted test reward": adapt_avg_rew,
                                                       "adapt/test reward": avg_rew}, run_name="")


    env2 = RepeatedBoSEnv(partner_policies=testing_partner, horizon=n_eval)
    all_rewards = []
    for _ in range(10):
        model2 = LTA_PPO.load("./data/tmp_lta_ppo")
        _, _, rewards = eval_model(model2, env2, n_eval=n_eval, adapt=True, 
            trained_steps=train_timesteps, log=False)
        all_rewards.append(rewards)

    nonadapt_rewards = []
    for _ in range(10):
        model2 = LTA_PPO.load("./data/tmp_lta_ppo")
        _, _, rewards = eval_model(model2, env2, n_eval=n_eval, adapt=False, 
            trained_steps=train_timesteps, log=False)
        nonadapt_rewards.append(rewards)

    vanilla_rewards = []
    if train_vanilla:
        for _ in range(10):
            _, _, rewards = eval_model(model_ppo, env2, n_eval=n_eval, adapt=False, 
                trained_steps=train_timesteps, log=False)
            vanilla_rewards.append(rewards)

    plt.plot(np.mean(all_rewards, axis=0), label="adapted")
    plt.plot(np.mean(nonadapt_rewards, axis=0), label="non-adapted")
    if train_vanilla:
        plt.plot(np.mean(vanilla_rewards, axis=0), label="vanilla ppo")
    plt.legend()
    plt.show()
