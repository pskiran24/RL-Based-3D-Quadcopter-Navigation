%% GRID SEARCH DOUBLE Q-LEARNING 3D DRONE PATH OPTIMIZATION - MULTIPLE OBSTACLES
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
learning_rates = [0.05, 0.1];
discount_factors = [0.95, 0.99];
epsilons = [0.2, 0.3];
% Create combinations of parameters
param_combinations = combvec(learning_rates, discount_factors, epsilons)';
num_combinations = size(param_combinations,1);
% To store results
all_episode_rewards = cell(num_combinations,1);
all_success_rates = zeros(num_combinations,1);
all_avg_rewards = zeros(num_combinations,1);
%% 3. PERFORM GRID SEARCH
best_avg_reward = -inf;
best_params = [];
best_Q1 = [];
best_Q2 = [];
for idx = 1:num_combinations
    lr = param_combinations(idx,1);
    df = param_combinations(idx,2);
    ep = param_combinations(idx,3);
    
    fprintf('Training with LR=%.3f DF=%.3f Epsilon=%.2f\n', lr, df, ep);
    
    % Initialize Q-tables BEFORE calling the function
    Q1_init = zeros(num_x_bins, num_y_bins, num_z_bins, num_actions);
    Q2_init = zeros(num_x_bins, num_y_bins, num_z_bins, num_actions);
    
    [Q1, Q2, episode_rewards, success_rate, avg_reward] = train_double_qlearning(lr, df, ep, num_episodes, max_steps_per_episode, ...
        Q1_init, Q2_init, drone_start, target_pos, ...
        actions, num_actions, ...
        grid_resolution, num_x_bins, num_y_bins, num_z_bins, ...
        env_size, env_height, num_obstacles);
    
    all_episode_rewards{idx} = episode_rewards;
    all_success_rates(idx) = success_rate;
    all_avg_rewards(idx) = avg_reward;
    
    if avg_reward > best_avg_reward
        best_avg_reward = avg_reward;
        best_params = [lr, df, ep];
        best_Q1 = Q1;
        best_Q2 = Q2;
        save(sprintf('best_DoubleQLEARNING_model_LR%.3f_DF%.3f_EP%.2f.mat', lr, df, ep), 'Q1', 'Q2');
    end
    
    fprintf('Avg Reward: %.2f, Success Rate: %.1f%%\n\n', avg_reward, success_rate);
end
%% 4. PLOT RESULTS - LEARNING CURVES PER HYPERPARAMETER
num_rows = ceil(sqrt(num_combinations));
num_cols = ceil(num_combinations / num_rows);
figure('Name', 'Double Q-Learning Learning Curves per Hyperparameter', 'Position', [100, 100, 1400, 900]);
for idx = 1:num_combinations
    subplot(num_rows, num_cols, idx);
    
    rewards = all_episode_rewards{idx};
    window_size = 100; % Smoothing window
    smoothed_rewards = movmean(rewards, window_size);
    
    plot(rewards, 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5);
    hold on;
    plot(smoothed_rewards, 'b-', 'LineWidth', 2);
    
    title(sprintf('LR=%.2f DF=%.2f ε=%.2f\nAvg Reward: %.1f | Success: %.1f%%', ...
        param_combinations(idx,1), param_combinations(idx,2), param_combinations(idx,3), ...
        all_avg_rewards(idx), all_success_rates(idx)), 'FontSize', 9);
    
    xlabel('Episode');
    ylabel('Total Reward');
    grid on;
    ylim([min(rewards)-500, max(rewards)+500]);
    
    if all_avg_rewards(idx) == best_avg_reward
        set(gca, 'LineWidth', 2, 'XColor', 'r', 'YColor', 'r');
    end
end
figure('Name', 'Double Q-Learning Hyperparameter Comparison', 'Position', [100, 100, 1200, 600]);
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
    legendStrings{idx} = sprintf('LR=%.2f DF=%.2f ε=%.2f', param_combinations(idx,:));
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
fprintf('%-4s | %-6s | %-6s | %-6s | %-12s | %-12s\n', 'Rank', 'LR', 'DF', 'Eps', 'Avg Reward', 'Success Rate');
fprintf('-----|--------|--------|--------|--------------|-------------\n');
[sorted_rewards, sort_idx] = sort(all_avg_rewards, 'descend');
for i = 1:num_combinations
    idx = sort_idx(i);
    fprintf('%-4d | %.4f | %.4f | %.4f | %12.2f | %11.1f%%\n', ...
        i, param_combinations(idx,1), param_combinations(idx,2), param_combinations(idx,3), ...
        all_avg_rewards(idx), all_success_rates(idx));
end
fprintf('===================================================\n\n');
fprintf('🏆 BEST DOUBLE Q-LEARNING MODEL:\n');
fprintf('   Learning Rate    = %.4f\n', best_params(1));
fprintf('   Discount Factor  = %.4f\n', best_params(2));
fprintf('   Epsilon          = %.4f\n', best_params(3));
fprintf('   Average Reward   = %.2f\n', best_avg_reward);
fprintf('   Success Rate     = %.1f%%\n', all_success_rates(best_idx));
fprintf('===================================================\n');
%% 5. TRAIN_DOUBLE_QLEARNING FUNCTION
function [Q1, Q2, episode_rewards, success_rate, avg_reward] = train_double_qlearning(learning_rate, discount_factor, epsilon, num_episodes, max_steps_per_episode, ...
    Q1, Q2, drone_start, target_pos, actions, num_actions, grid_resolution, num_x_bins, num_y_bins, num_z_bins, env_size, env_height, num_obstacles)
    episode_rewards = zeros(num_episodes,1);
    successful_episodes = 0;
    for episode = 1:num_episodes
        obstacles = generate_random_obstacles(num_obstacles, env_size, env_height);
        drone_pos = drone_start;
        
        [x_bin, y_bin, z_bin] = get_state_bins(drone_pos, grid_resolution, num_x_bins, num_y_bins, num_z_bins);
        
        total_reward = 0;
        for step = 1:max_steps_per_episode
            % Select action using epsilon-greedy policy on combined Q
            Q_sum = Q1(x_bin, y_bin, z_bin, :) + Q2(x_bin, y_bin, z_bin, :);
            if rand() < epsilon
                action_idx = randi(num_actions);
            else
                [~, action_idx] = max(Q_sum);
            end
            
            % Execute action
            new_pos = drone_pos + actions{action_idx};
            if ~check_bounds(new_pos, env_size, env_height) || check_collision(new_pos, obstacles)
                new_pos = drone_pos; % stay
            end
            
            % Calculate reward
            reward = calculate_reward(new_pos, drone_pos, target_pos, obstacles, env_size, env_height);
            total_reward = total_reward + reward;
            
            % Get next state
            [new_x_bin, new_y_bin, new_z_bin] = get_state_bins(new_pos, grid_resolution, num_x_bins, num_y_bins, num_z_bins);
            
            % Decide randomly which Q-table to update
            if rand() < 0.5
                % Update Q1
                [~, best_next_action] = max(Q1(new_x_bin, new_y_bin, new_z_bin, :));
                target_Q = Q2(new_x_bin, new_y_bin, new_z_bin, best_next_action);
                old_Q = Q1(x_bin, y_bin, z_bin, action_idx);
                Q1(x_bin, y_bin, z_bin, action_idx) = old_Q + learning_rate * (reward + discount_factor * target_Q - old_Q);
            else
                % Update Q2
                [~, best_next_action] = max(Q2(new_x_bin, new_y_bin, new_z_bin, :));
                target_Q = Q1(new_x_bin, new_y_bin, new_z_bin, best_next_action);
                old_Q = Q2(x_bin, y_bin, z_bin, action_idx);
                Q2(x_bin, y_bin, z_bin, action_idx) = old_Q + learning_rate * (reward + discount_factor * target_Q - old_Q);
            end
            
            % Move to next state
            drone_pos = new_pos;
            x_bin = new_x_bin; y_bin = new_y_bin; z_bin = new_z_bin;
            
            % Check if goal reached
            if norm(drone_pos - target_pos) < 5
                successful_episodes = successful_episodes + 1;
                break;
            end
        end
        episode_rewards(episode) = total_reward;
    end
    success_rate = (successful_episodes / num_episodes) * 100;
    avg_reward = mean(episode_rewards);
end

%% 6. HELPER FUNCTIONS (same as your Q-learning code)
function obstacles = generate_random_obstacles(num_obs, env_size, env_height)
    obstacles = zeros(num_obs, 6);
    for i = 1:num_obs
        x = rand() * (env_size - 20) + 10;  
        y = rand() * (env_size - 20) + 10;
        z = rand() * (env_height - 20) + 10;
        size_val = rand() * 5 + 10; % 10 to 15 meters
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
        reward = 1;  % Moving closer
    else
        reward = -0.5;  % Moving away
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

%% 7. VISUALIZE BEST MODEL PATH
fprintf('\nVisualizing best Double Q-Learning model...\n');
obstacles_vis = generate_random_obstacles(num_obstacles, env_size, env_height);
drone_pos = drone_start;
path = drone_pos;
epsilon_greedy = 0; % No exploration for visualization
for step = 1:max_steps_per_episode
    [x_bin, y_bin, z_bin] = get_state_bins(drone_pos, grid_resolution, num_x_bins, num_y_bins, num_z_bins);
    % Use combined Q to select action greedily
    Q_sum = best_Q1(x_bin, y_bin, z_bin, :) + best_Q2(x_bin, y_bin, z_bin, :);
    [~, action_idx] = max(Q_sum);
    
    new_pos = drone_pos + actions{action_idx};
    if ~check_bounds(new_pos, env_size, env_height) || check_collision(new_pos, obstacles_vis)
        new_pos = drone_pos; % stay if collision or out of bounds
    end
    
    drone_pos = new_pos;
    path = [path; drone_pos];
    
    if norm(drone_pos - target_pos) < 5
        fprintf('Target reached in %d steps!\n', step);
        break;
    end
end
figure('Name', 'Best Double Q-Learning Model - 3D Path Visualization');
hold on;
grid on;
axis equal;
xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
title(sprintf('Best Double Q-Learning Path (LR=%.3f, DF=%.3f, Eps=%.2f)', best_params));
xlim([0 env_size]);
ylim([0 env_size]);
zlim([0 env_height]);
view(3);
for i = 1:size(obstacles_vis, 1)
    plot_obstacle(obstacles_vis(i, :));
end
plot3(drone_start(1), drone_start(2), drone_start(3), 'go', 'MarkerSize', 15, 'MarkerFaceColor', 'g', 'DisplayName', 'Start');
plot3(target_pos(1), target_pos(2), target_pos(3), 'r^', 'MarkerSize', 20, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k', 'DisplayName', 'Target', 'LineWidth', 3);
[xs, ys, zs] = sphere(20);
surf(xs*3 + target_pos(1), ys*3 + target_pos(2), zs*3 + target_pos(3), 'FaceColor', 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
plot3(path(:,1), path(:,2), path(:,3), 'b-', 'LineWidth', 2, 'DisplayName', 'Drone Path');
plot3(path(:,1), path(:,2), path(:,3), 'bo', 'MarkerSize', 4, 'MarkerFaceColor', 'b');
legend('Location', 'best');
hold off;

%% 8. HELPER FUNCTION FOR OBSTACLE PLOTTING
function plot_obstacle(obs)
    center = obs(1:3);
    sizes = obs(4:6);
    x_range = center(1) + [-sizes(1)/2, sizes(1)/2];
    y_range = center(2) + [-sizes(2)/2, sizes(2)/2];
    z_range = center(3) + [-sizes(3)/2, sizes(3)/2];
    [X, Y, Z] = meshgrid(x_range, y_range, z_range);
    % Bottom and top faces
    surf([X(1,1,1) X(1,2,1); X(2,1,1) X(2,2,1)], ...
         [Y(1,1,1) Y(1,1,1); Y(2,1,1) Y(2,1,1)], ...
         [Z(1,1,1) Z(1,1,1); Z(1,1,1) Z(1,1,1)], 'FaceColor', 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'k');
    surf([X(1,1,2) X(1,2,2); X(2,1,2) X(2,2,2)], ...
         [Y(1,1,2) Y(1,1,2); Y(2,1,2) Y(2,1,2)], ...
         [Z(1,1,2) Z(1,1,2); Z(1,1,2) Z(1,1,2)], 'FaceColor', 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'k');
    % Front and back faces
    surf([X(1,1,1) X(1,2,1); X(1,1,2) X(1,2,2)], ...
         [Y(1,1,1) Y(1,1,1); Y(1,1,2) Y(1,1,2)], ...
         [Z(1,1,1) Z(1,1,1); Z(1,1,2) Z(1,1,2)], 'FaceColor', 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'k');
    surf([X(2,1,1) X(2,2,1); X(2,1,2) X(2,2,2)], ...
         [Y(2,1,1) Y(2,1,1); Y(2,1,2) Y(2,1,2)], ...
         [Z(2,1,1) Z(2,1,1); Z(2,1,2) Z(2,1,2)], 'FaceColor', 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'k');
    % Left and right faces
    surf([X(1,1,1) X(1,1,2); X(2,1,1) X(2,1,2)], ...
         [Y(1,1,1) Y(2,1,1); Y(1,1,2) Y(2,1,2)], ...
         [Z(1,1,1) Z(1,1,1); Z(1,1,2) Z(1,1,2)], 'FaceColor', 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'k');
    surf([X(1,2,1) X(1,2,2); X(2,2,1) X(2,2,2)], ...
         [Y(1,2,1) Y(2,2,1); Y(1,2,2) Y(2,2,2)], ...
         [Z(1,2,1) Z(1,2,1); Z(1,2,2) Z(1,2,2)], 'FaceColor', 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'k');
end
