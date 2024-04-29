import numpy as np
import matplotlib.pyplot as plt
import torch

def mppi(func_control_update_converged,
         func_comp_weights, func_term_cost, func_run_cost, func_g, func_F,
         func_state_transform, func_filter_du, num_samples, learning_rate,
         init_state, init_ctrl_seq, ctrl_noise_covar, time_horizon,
         per_ctrl_based_ctrl_noise, real_traj_cost, print_mppi):

    # time stuff
    control_dim, num_timesteps = init_ctrl_seq.size()
    dt = time_horizon / num_timesteps

    # sample state stuff
    sample_init_state = func_state_transform(init_state)
    sample_state_dim = sample_init_state.size()[0]

    # state trajectories
    real_x_traj = torch.zeros(sample_state_dim, num_timesteps + 1)
    real_x_traj[:, 0:1] = sample_init_state

    x_traj = torch.zeros(sample_state_dim, num_samples, num_timesteps + 1)
    x_traj[:, :, 0] = sample_init_state.repeat(1, num_samples)

    # control stuff
    du = torch.ones(control_dim, num_timesteps) * 1e6

    # control sequence
    sample_u_traj = init_ctrl_seq

    # sampled control trajectories
    v_traj = torch.zeros(control_dim, num_samples, num_timesteps)

    # Begin mppi
    iteration = 1
    while not func_control_update_converged(du, iteration):
        last_sample_u_traj = sample_u_traj.clone().detach()

        # Noise generation
        flat_distribution = torch.normal(torch.zeros(control_dim, num_samples * num_timesteps))
        ctrl_noise_flat = ctrl_noise_covar @ flat_distribution
        ctrl_noise = torch.reshape(ctrl_noise_flat, (control_dim, num_samples, num_timesteps))

        # Compute sampled control trajectories
        # The number of trajectories that have both control and noise
        ctrl_based_ctrl_noise_samples = int(np.round(per_ctrl_based_ctrl_noise * num_samples))

        if ctrl_based_ctrl_noise_samples == 0:
            v_traj = ctrl_noise
        elif ctrl_based_ctrl_noise_samples == num_samples:
            v_traj = sample_u_traj.view(control_dim, 1, num_timesteps) + ctrl_noise
        else:
            v_traj[:, :ctrl_based_ctrl_noise_samples, :] = sample_u_traj.view(
                control_dim, 1, num_timesteps) + ctrl_noise[:, :ctrl_based_ctrl_noise_samples, :]
            v_traj[:, ctrl_based_ctrl_noise_samples:, :] = ctrl_noise[:, ctrl_based_ctrl_noise_samples:, :]

        # Forward propagation #sample trajectories
        for timestep_num in range(num_timesteps):
            x_traj[:, :, timestep_num+1] = func_F(x_traj[:, :, timestep_num], func_g(v_traj[:, :, timestep_num]), dt)
            if print_mppi:
                print("TN: {}, IN: {}, DU: {}".format(timestep_num, iteration, torch.mean(torch.sum(torch.abs(du), dim=0))))

        traj_cost = torch.zeros(1, num_samples)
        for timestep_num in range(num_timesteps):
            traj_cost = (traj_cost + func_run_cost(x_traj[:, :, timestep_num]) +
                         learning_rate * torch.t(sample_u_traj[:, timestep_num])  @  ctrl_noise_covar.inverse() @ (sample_u_traj[:, timestep_num] - v_traj[:, :, timestep_num]))
        traj_cost = traj_cost + func_term_cost(x_traj[:, :, timestep_num])

        # Weight and du calculation
        w = func_comp_weights(traj_cost)
        du = torch.sum(w.view(control_dim, num_samples, 1) * ctrl_noise, dim=1)  # [control_dim, num_timesteps]

        # Filter the output from forward propagation
        du = func_filter_du(du)

        sample_u_traj = sample_u_traj + du
        iteration += 1

    # TODO(cgliu): probably we don't need to do this.
    # Weight and du calculation
    w = func_comp_weights(traj_cost)
    du = torch.sum(w.view(control_dim, num_samples, 1) * ctrl_noise, dim=1)  # [control_dim, num_timesteps]

    # Filter the output from forward propagation
    du = func_filter_du(du)
    sample_u_traj = sample_u_traj + du
    iteration = iteration + 1

    if real_traj_cost:
        # Loop through the dynamics again to recalcuate traj_cost
        rep_traj_cost = 0
        # Forward propagation
        for timestep_num in range(num_timesteps):
            real_x_traj[:, timestep_num+1:timestep_num+2] = func_F(real_x_traj[:, timestep_num:timestep_num+1],
                                                                   func_g(sample_u_traj[:, timestep_num:timestep_num+1]), dt)

        for timestep_num in range(num_timesteps):
            rep_traj_cost = (rep_traj_cost + func_run_cost(real_x_traj[:, timestep_num:timestep_num+1]) +
                             learning_rate * sample_u_traj[:, timestep_num:timestep_num+1].T @ ctrl_noise_covar.inverse() @ (last_sample_u_traj[:, timestep_num:timestep_num+1] - sample_u_traj[:, timestep_num:timestep_num+1]))
        rep_traj_cost = rep_traj_cost + func_term_cost(real_x_traj[:, timestep_num:timestep_num+1])
    else:
        # normalize weights, in case they are not normalized
        normalized_w = w / torch.sum(w)  # todo() necessary??

        # Compute the representative trajectory cost of what actually happens
        # another way to think about this is weighted average of sample trajectory costs
        rep_traj_cost = torch.sum(normalized_w * traj_cost)

    return sample_u_traj, rep_traj_cost.item(), real_x_traj
