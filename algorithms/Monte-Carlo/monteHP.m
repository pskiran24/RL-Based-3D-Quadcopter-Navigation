%% GRID SEARCH MONTE CARLO CONTROL 3D DRONE PATH OPTIMIZATION - MULTIPLE OBSTACLES
clear all; close all; clc;
%% 1. ENVIRONMENT SETUP
env_size = 100; % meters
env_height = 50; % meters
drone_start = [10, 10, 10];
target_pos = [80, 80, 30];
num_obstacles = 5;
%% 2. RL PARAMETERS AND GRID SEARCH SETUP
step_size = 2; % meters per action
actions = {
    [-step_size, 0, 0],        % Left
    [step_size, 0, 0],         % Right
    [0, step_size, 0],         % Forward
    [0, -step_size, 0],        % Backward
    [0, 0, step_size],         % Up
    [0, 0, -step_size]         % Down
};
num_actions = length(actions);
grid_resolution = 5;
num_x_bins = ceil(env_size/grid_resolution);
num_y_bins = ceil(env_size/grid_resolution);
num_z_bins = ceil(env_height/grid_resolution);
max_steps_per_episode = 200;
num_episodes = 5000;
% Hyperparameter ranges
learning_rates = [0.05, 0.1]; % Note: MC usually does not use learning rate, kept for consistency
discount_factors = [0.95, 0.99];
epsilons = [0.2, 0.3];
param_combinations = combvec(learning_rates, discount_factors, epsilons)';
num_combinations = size(param_combinations,1);
% To store results
all_episode_rewards = cell(num_combinations,1);
all_success_rates = zeros(num_combinations,1);
all_avg_rewards = zeros(num_combinations,1);
%% 3. PERFORM GRID SEARCH
best_avg_reward = -inf;
best_params = [];
best_Q = [];
N = []; % Visit counts
for idx = 1:num_combinations
    lr = param_combinations(idx,1); % not used directly but kept optional
    df = param_combinations(idx,2);
    ep = param_combinations(idx,3);
    
    fprintf('Training with DF=%.3f Epsilon=%.2f (Note: LR ignored in MC)\n', df, ep);
    
    % Initialize Q-table and visit counts
    Q_init = zeros(num_x_bins, num_y_bins, num_z_bins, num_actions);
    N_init = zeros(num_x_bins, num_y_bins, num_z_bins, num_actions); % counts for averaging returns
    
    [Q, N, episode_rewards, success_rate, avg_reward] = train_monte_carlo(df, ep, num_episodes, max_steps_per_episode, ...
        Q_init, N_init, drone_start, target_pos, ...
        actions, num_actions, ...
        grid_resolution, num_x_bins, num_y_bins, num_z_bins, ...
        env_size, env_height, num_obstacles);
    
    all_episode_rewards{idx} = episode_rewards;
    all_success_rates(idx) = success_rate;
    all_avg_rewards(idx) = avg_reward;
    
    if avg_reward > best_avg_reward
        best_avg_reward = avg_reward;
        best_params = [lr, df, ep];
        best_Q = Q;
        save(sprintf('best_MC_model_DF%.3f_EP%.2f.mat', df, ep), 'Q', 'N');
    end
    
    fprintf('Avg Reward: %.2f, Success Rate: %.1f%%\n\n', avg_reward, success_rate);
end
%% 4. PLOT RESULTS - LEARNING CURVES PER HYPERPARAMETER
num_rows = ceil(sqrt(num_combinations));
num_cols = ceil(num_combinations / num_rows);
figure('Name', 'Monte Carlo Learning Curves per Hyperparameter', 'Position', [100, 100, 1400, 900]);
for idx = 1:num_combinations
    subplot(num_rows, num_cols, idx);
    
    rewards = all_episode_rewards{idx};
    window_size = 100; % Smoothing window
    smoothed_rewards = movmean(rewards, window_size);
    
    plot(rewards, 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5);
    hold on;
    plot(smoothed_rewards, 'b-', 'LineWidth', 2);
    
    title(sprintf('DF=%.2f ε=%.2f\nAvg Reward: %.1f | Success: %.1f%%', ...
        param_combinations(idx,2), param_combinations(idx,3), ...
        all_avg_rewards(idx), all_success_rates(idx)), 'FontSize', 9);
    
    xlabel('Episode');
    ylabel('Total Reward');
    grid on;
    ylim([min(rewards)-500, max(rewards)+500]);
    
    if all_avg_rewards(idx) == best_avg_reward
        set(gca, 'LineWidth', 2, 'XColor', 'r', 'YColor', 'r');
    end
end
figure('Name', 'Monte Carlo Hyperparameter Comparison', 'Position', [100, 100, 1200, 600]);
subplot(1, 2, 1);
bar(all_avg_rewards, 'FaceColor', [0.2 0.6 0.8]);
hold on;
[~, best_idx] = max(all_avg_rewards);
bar(best_idx, all_avg_rewards(best_idx), 'FaceColor', [0.8 0.2 0.2]);
ylabel('Average Reward', 'FontSize', 12, 'FontWeight', 'bold');
title('Average Reward Comparison', 'FontSize', 14);
grid on;
legendStrings = cell(num_combinations, 1);
for idx = 1:num_combinations
    legendStrings{idx} = sprintf('DF=%.2f ε=%.2f', param_combinations(idx,2), param_combinations(idx,3));
end
set(gca, 'XTick', 1:num_combinations, 'XTickLabel', 1:num_combinations);
xtickangle(45);
xlabel('Configuration #');
for idx = 1:num_combinations
    text(idx, all_avg_rewards(idx), sprintf('%.1f', all_avg_rewards(idx)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 9);
end
subplot(1, 2, 2);
bar(all_success_rates, 'FaceColor', [0.3 0.7 0.4]);
hold on;
bar(best_idx, all_success_rates(best_idx), 'FaceColor', [0.8 0.2 0.2]);
ylabel('Success Rate (%)', 'FontSize', 12, 'FontWeight', 'bold');
title('Success Rate Comparison', 'FontSize', 14);
grid on;
set(gca, 'XTick', 1:num_combinations, 'XTickLabel', 1:num_combinations);
xtickangle(45);
xlabel('Configuration #');
ylim([0 100]);
for idx = 1:num_combinations
    text(idx, all_success_rates(idx), sprintf('%.1f%%', all_success_rates(idx)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 9);
end
fprintf('\n========== HYPERPARAMETER TUNING RESULTS ==========\n');
fprintf('%-4s | %-6s | %-6s | %-12s | %-12s\n', 'Rank', 'DF', 'Eps', 'Avg Reward', 'Success Rate');
fprintf('-----|--------|--------|--------------|-------------\n');
[sorted_rewards, sort_idx] = sort(all_avg_rewards, 'descend');
for i = 1:num_combinations
    idx = sort_idx(i);
    fprintf('%-4d | %.4f | %.4f | %12.2f | %11.1f%%\n', ...
        i, param_combinations(idx,2), param_combinations(idx,3), ...
        all_avg_rewards(idx), all_success_rates(idx));
end
fprintf('===================================================\n\n');
fprintf('🏆 BEST MONTE CARLO MODEL:\n');
fprintf('   Discount Factor  = %.4f\n', best_params(2));
fprintf('   Epsilon          = %.4f\n', best_params(3));
fprintf('   Average Reward   = %.2f\n', best_avg_reward);
fprintf('   Success Rate     = %.1f%%\n', all_success_rates(best_idx));
fprintf('===================================================\n');
%% 5. TRAIN_MONTE_CARLO FUNCTION
function [Q, N, episode_rewards, success_rate, avg_reward] = train_monte_carlo(discount_factor, epsilon, num_episodes, max_steps_per_episode, ...
    Q, N, drone_start, target_pos, actions, num_actions, grid_resolution, num_x_bins, num_y_bins, num_z_bins, env_size, env_height, num_obstacles)
    episode_rewards = zeros(num_episodes,1);
    successful_episodes = 0;
    for episode = 1:num_episodes
        obstacles = generate_random_obstacles(num_obstacles, env_size, env_height);
        drone_pos = drone_start;
        
        states = zeros(max_steps_per_episode, 3);
        actions_taken = zeros(max_steps_per_episode,1);
        rewards = zeros(max_steps_per_episode,1);
        
        % Generate episode (using epsilon-greedy policy on current Q)
        for step = 1:max_steps_per_episode
            [x_bin, y_bin, z_bin] = get_state_bins(drone_pos, grid_resolution, num_x_bins, num_y_bins, num_z_bins);
            if rand() < epsilon
                action_idx = randi(num_actions);
            else
                [~, action_idx] = max(Q(x_bin, y_bin, z_bin, :));
            end
            new_pos = drone_pos + actions{action_idx};
            if ~check_bounds(new_pos, env_size, env_height) || check_collision(new_pos, obstacles)
                new_pos = drone_pos; % stay in place
            end
            reward = calculate_reward(new_pos, drone_pos, target_pos, obstacles, env_size, env_height);
            
            states(step, :) = [x_bin, y_bin, z_bin];
            actions_taken(step) = action_idx;
            rewards(step) = reward;
            
            drone_pos = new_pos;
            
            if norm(drone_pos - target_pos) < 5
                step = step; % end episode early
                successful_episodes = successful_episodes + 1;
                break;
            end
        end
        % Calculate returns and update Q after episode (first-visit MC)
        G = 0;
        visited = zeros(num_x_bins, num_y_bins, num_z_bins, num_actions);
        for t = step:-1:1
            G = discount_factor * G + rewards(t);
            s = states(t,:);
            a = actions_taken(t);
            if visited(s(1), s(2), s(3), a) == 0
                visited(s(1), s(2), s(3), a) = 1;
                N(s(1), s(2), s(3), a) = N(s(1), s(2), s(3), a) + 1;
                alpha = 1 / N(s(1), s(2), s(3), a);
                Q(s(1), s(2), s(3), a) = Q(s(1), s(2), s(3), a) + alpha * (G - Q(s(1), s(2), s(3), a));
            end
        end
        episode_rewards(episode) = sum(rewards(1:step));
    end
    success_rate = (successful_episodes / num_episodes) * 100;
    avg_reward = mean(episode_rewards);
end
%% 6. HELPER FUNCTIONS (same as before)
function obstacles = generate_random_obstacles(num_obs, env_size, env_height)
    obstacles = zeros(num_obs, 6);
    for i = 1:num_obs
        x = rand() * (env_size - 20) + 10;  
        y = rand() * (env_size - 20) + 10;
        z = rand() * (env_height - 20) + 10;
        size_val = rand() * 5 + 10;
        obstacles(i, :) = [x, y, z, size_val, size_val, size_val];
    end
end
function [x_bin, y_bin, z_bin] = get_state_bins(pos, grid_resolution, num_x_bins, num_y_bins, num_z_bins)
    x_bin = min(max(ceil(pos(1) / grid_resolution), 1), num_x_bins);
    y_bin = min(max(ceil(pos(2) / grid_resolution), 1), num_y_bins);
    z_bin = min(max(ceil(pos(3) / grid_resolution), 1), num_z_bins);
end
function collision = check_collision(pos, obstacles)
    collision = false;
    for i = 1:size(obstacles, 1)
        obs_center = obstacles(i, 1:3);
        obs_size = obstacles(i, 4:6);
        if all(abs(pos - obs_center) <= obs_size / 2)
            collision = true;
            return;
        end
    end
end
function valid = check_bounds(pos, env_size, env_height)
    valid = pos(1) >= 0 && pos(1) <= env_size && ...
            pos(2) >= 0 && pos(2) <= env_size && ...
            pos(3) >= 0 && pos(3) <= env_height;
end
function reward = calculate_reward(new_pos, old_pos, target_pos, obstacles, env_size, env_height)
    old_dist = norm(old_pos - target_pos);
    new_dist = norm(new_pos - target_pos);
    if new_dist < old_dist
        reward = 1;
    else
        reward = -0.5;
    end
    if check_collision(new_pos, obstacles)
        reward = reward - 100;
    end
    if ~check_bounds(new_pos, env_size, env_height)
        reward = reward - 100;
    end
    if new_dist < 5
        reward = reward + 100;
    end
end
