%% SARSA TESTING WITH ENHANCED VISUALIZATION - 100 EPISODES
clear all; close all; clc;

%% 1. LOAD BEST MODEL
% Replace with your actual best model filename
model_file = 'best_SARSA_model_LR0.100_DF0.950_EP0.30.mat'; % Adjust this
if ~isfile(model_file)
    error('Model file not found! Please provide the correct filename.');
end
load(model_file, 'Q');
fprintf('✓ Loaded model: %s\n', model_file);

%% 2. ENVIRONMENT SETUP (Same as training)
env_size = 100; % meters
env_height = 50; % meters
drone_start = [10, 10, 10];
target_pos = [80, 80, 30];
num_obstacles = 5;
step_size = 2;

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

%% 3. TEST FOR 100 EPISODES
num_test_episodes = 100;
test_rewards = zeros(num_test_episodes, 1);
test_steps = zeros(num_test_episodes, 1);
test_success = zeros(num_test_episodes, 1);
all_paths = cell(num_test_episodes, 1);
all_obstacles = cell(num_test_episodes, 1);

fprintf('\n========== TESTING STARTED ==========\n');
epsilon_test = 0; % Greedy policy (no exploration)

for episode = 1:num_test_episodes
    % Generate random obstacles for this episode
    obstacles = generate_random_obstacles(num_obstacles, env_size, env_height);
    all_obstacles{episode} = obstacles;
    
    drone_pos = drone_start;
    path = drone_pos;
    total_reward = 0;
    
    for step = 1:max_steps_per_episode
        [x_bin, y_bin, z_bin] = get_state_bins(drone_pos, grid_resolution, num_x_bins, num_y_bins, num_z_bins);
        action_idx = select_action(Q, x_bin, y_bin, z_bin, epsilon_test, num_actions);
        
        new_pos = drone_pos + actions{action_idx};
        if ~check_bounds(new_pos, env_size, env_height) || check_collision(new_pos, obstacles)
            new_pos = drone_pos; % stay
        end
        
        reward = calculate_reward(new_pos, drone_pos, target_pos, obstacles, env_size, env_height);
        total_reward = total_reward + reward;
        
        drone_pos = new_pos;
        path = [path; drone_pos];
        
        % Check if reached target
        if norm(drone_pos - target_pos) < 5
            test_success(episode) = 1;
            test_steps(episode) = step;
            break;
        end
        test_steps(episode) = step;
    end
    
    test_rewards(episode) = total_reward;
    all_paths{episode} = path;
    
    if mod(episode, 10) == 0
        fprintf('Episode %d/100 completed\n', episode);
    end
end

%% 4. CALCULATE STATISTICS
success_rate = (sum(test_success) / num_test_episodes) * 100;
avg_reward = mean(test_rewards);
avg_steps = mean(test_steps(test_success == 1)); % Average steps for successful episodes
std_reward = std(test_rewards);

fprintf('\n========== TEST RESULTS ==========\n');
fprintf('Total Episodes:       %d\n', num_test_episodes);
fprintf('Successful Episodes:  %d\n', sum(test_success));
fprintf('Success Rate:         %.2f%%\n', success_rate);
fprintf('Average Reward:       %.2f ± %.2f\n', avg_reward, std_reward);
fprintf('Average Steps (Success): %.2f\n', avg_steps);
fprintf('===================================\n\n');

%% 5. PLOT TEST RESULTS
figure('Name', 'SARSA Test Results', 'Position', [100, 100, 1400, 600]);

% Subplot 1: Episode Rewards
subplot(2, 3, 1);
plot(test_rewards, 'b-', 'LineWidth', 1.5);
hold on;
plot(find(test_success), test_rewards(test_success == 1), 'go', 'MarkerSize', 6, 'MarkerFaceColor', 'g');
plot(find(~test_success), test_rewards(test_success == 0), 'rx', 'MarkerSize', 8, 'LineWidth', 2);
xlabel('Episode');
ylabel('Total Reward');
title('Test Episode Rewards');
legend('All Episodes', 'Success', 'Failure', 'Location', 'best');
grid on;

% Subplot 2: Success/Failure Distribution
subplot(2, 3, 2);
bar([sum(test_success), sum(~test_success)]);
set(gca, 'XTickLabel', {'Success', 'Failure'});
ylabel('Count');
title(sprintf('Success Rate: %.1f%%', success_rate));
grid on;
for i = 1:2
    val = [sum(test_success), sum(~test_success)];
    text(i, val(i), sprintf('%d', val(i)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end

% Subplot 3: Steps Distribution (for successful episodes)
subplot(2, 3, 3);
histogram(test_steps(test_success == 1), 20, 'FaceColor', 'g');
xlabel('Steps to Target');
ylabel('Frequency');
title('Steps Distribution (Successful Episodes)');
grid on;

% Subplot 4: Reward Distribution
subplot(2, 3, 4);
histogram(test_rewards, 30, 'FaceColor', 'b');
xlabel('Total Reward');
ylabel('Frequency');
title('Reward Distribution');
grid on;

% Subplot 5: Cumulative Success Rate
subplot(2, 3, 5);
cumulative_success = cumsum(test_success) ./ (1:num_test_episodes)' * 100;
plot(cumulative_success, 'b-', 'LineWidth', 2);
xlabel('Episode');
ylabel('Cumulative Success Rate (%)');
title('Cumulative Success Rate Over Episodes');
grid on;
ylim([0 100]);

% Subplot 6: Summary Text
subplot(2, 3, 6);
axis off;
summary_text = sprintf(['TEST SUMMARY\n\n' ...
    'Episodes: %d\n' ...
    'Success Rate: %.1f%%\n' ...
    'Avg Reward: %.2f\n' ...
    'Std Reward: %.2f\n' ...
    'Avg Steps (Success): %.1f\n' ...
    'Max Reward: %.2f\n' ...
    'Min Reward: %.2f'], ...
    num_test_episodes, success_rate, avg_reward, std_reward, avg_steps, ...
    max(test_rewards), min(test_rewards));
text(0.1, 0.5, summary_text, 'FontSize', 12, 'FontWeight', 'bold', 'VerticalAlignment', 'middle');

%% 6. VISUALIZE SAMPLE EPISODES WITH ENHANCED DRONE MODELS
% Find best, worst, and a random episode
[~, best_idx] = max(test_rewards);
[~, worst_idx] = min(test_rewards);
random_idx = randi(num_test_episodes);

sample_episodes = [best_idx, worst_idx, random_idx];
sample_titles = {'Best Episode', 'Worst Episode', 'Random Episode'};

for i = 1:3
    ep_idx = sample_episodes(i);
    
    figure('Name', sprintf('%s (Episode %d)', sample_titles{i}, ep_idx), 'Position', [100 + (i-1)*400, 100, 800, 700]);
    
    % 3D Visualization with Enhanced Drone Model
    subplot(2, 1, 1);
    ax = gca;
    hold on;
    grid on;
    axis equal;
    xlabel('X (m)');
    ylabel('Y (m)');
    zlabel('Z (m)');
    title(sprintf('%s - Reward: %.1f, Steps: %d, Success: %s', ...
        sample_titles{i}, test_rewards(ep_idx), test_steps(ep_idx), ...
        string(test_success(ep_idx) == 1)));
    xlim([0 env_size]);
    ylim([0 env_size]);
    zlim([0 env_height]);
    view(45, 30);
    
    % Plot obstacles
    obstacles_ep = all_obstacles{ep_idx};
    for j = 1:size(obstacles_ep, 1)
        plot_obstacle(ax, obstacles_ep(j, 1:3), obstacles_ep(j, 4:6));
    end
    
    % Plot target as green star
    plot3(target_pos(1), target_pos(2), target_pos(3), 'g*', 'MarkerSize', 20, 'LineWidth', 3, 'DisplayName', 'Target');
    
    % Plot path
    path_ep = all_paths{ep_idx};
    if test_success(ep_idx)
        plot3(path_ep(:,1), path_ep(:,2), path_ep(:,3), 'b-', 'LineWidth', 2.5, 'DisplayName', 'Path (Success)');
        plot3(path_ep(:,1), path_ep(:,2), path_ep(:,3), 'bo', 'MarkerSize', 4, 'MarkerFaceColor', 'b');
    else
        plot3(path_ep(:,1), path_ep(:,2), path_ep(:,3), 'r-', 'LineWidth', 2.5, 'DisplayName', 'Path (Failed)');
        plot3(path_ep(:,1), path_ep(:,2), path_ep(:,3), 'ro', 'MarkerSize', 4, 'MarkerFaceColor', 'r');
    end
    
    % Plot start position with enhanced drone model
    drone_start_model = create_drone_model(ax, drone_start, 'green');
    
    % Plot final drone position with enhanced drone model
    final_pos = path_ep(end, :);
    if test_success(ep_idx)
        drone_final_model = create_drone_model(ax, final_pos, 'blue');
    else
        drone_final_model = create_drone_model(ax, final_pos, 'red');
    end
    
    legend('Location', 'best');
    hold off;
    
    % 2D Top-Down View
    subplot(2, 1, 2);
    hold on;
    grid on;
    axis equal;
    xlabel('X (m)');
    ylabel('Y (m)');
    title('Top-Down View (X-Y Plane)');
    xlim([0 env_size]);
    ylim([0 env_size]);
    
    % Plot obstacles (top view)
    for j = 1:size(obstacles_ep, 1)
        obs = obstacles_ep(j, :);
        rectangle('Position', [obs(1)-obs(4)/2, obs(2)-obs(5)/2, obs(4), obs(5)], ...
            'FaceColor', [1 0 0 0.3], 'EdgeColor', 'k', 'LineWidth', 1);
    end
    
    % Plot start and target
    plot(drone_start(1), drone_start(2), 'go', 'MarkerSize', 15, 'MarkerFaceColor', 'g');
    plot(target_pos(1), target_pos(2), 'g*', 'MarkerSize', 15, 'LineWidth', 2);
    
    % Plot path
    if test_success(ep_idx)
        plot(path_ep(:,1), path_ep(:,2), 'b-', 'LineWidth', 2);
        plot(path_ep(:,1), path_ep(:,2), 'bo', 'MarkerSize', 4, 'MarkerFaceColor', 'b');
    else
        plot(path_ep(:,1), path_ep(:,2), 'r-', 'LineWidth', 2);
        plot(path_ep(:,1), path_ep(:,2), 'ro', 'MarkerSize', 4, 'MarkerFaceColor', 'r');
    end
    
    hold off;
end

%% 7. HELPER FUNCTIONS
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

function action_idx = select_action(Q, x_bin, y_bin, z_bin, epsilon, num_actions)
    if rand() < epsilon
        action_idx = randi(num_actions);
    else
        [~, action_idx] = max(Q(x_bin, y_bin, z_bin, :));
    end
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

function drone_model = create_drone_model(ax, pos, color_type)
    % Enhanced drone visualization with body and rotors
    if nargin < 3
        color_type = 'blue';
    end
    
    % Set colors based on type
    if strcmp(color_type, 'blue')
        body_color = [0.2, 0.2, 0.8];
        edge_color = 'blue';
    elseif strcmp(color_type, 'red')
        body_color = [0.8, 0.2, 0.2];
        edge_color = 'red';
    else % green
        body_color = [0.2, 0.8, 0.2];
        edge_color = 'green';
    end
    
    % Create drone body (rectangular prism)
    body_size = 1.5;
    
    body_vertices = [
        -body_size/2, -body_size/2, -body_size/4;
        body_size/2, -body_size/2, -body_size/4;
        body_size/2, body_size/2, -body_size/4;
        -body_size/2, body_size/2, -body_size/4;
        -body_size/2, -body_size/2, body_size/4;
        body_size/2, -body_size/2, body_size/4;
        body_size/2, body_size/2, body_size/4;
        -body_size/2, body_size/2, body_size/4;
    ];
    body_vertices = body_vertices + pos;
    
    body_faces = [1 2 3 4; 5 6 7 8; 1 2 6 5; 3 4 8 7; 1 4 8 5; 2 3 7 6];
    
    drone_model.body = patch(ax, 'Faces', body_faces, 'Vertices', body_vertices, ...
        'FaceColor', body_color, 'EdgeColor', edge_color, 'FaceAlpha', 0.8);
    
    % Create drone arms and rotors
    arm_length = 5;
    rotor_radius = 1;
    
    % Arm 1 (right-front)
    arm1 = [0, -2, 0; arm_length, -2, 0] + pos;
    drone_model.arm1 = plot3(ax, arm1(:,1), arm1(:,2), arm1(:,3), 'Color', edge_color, 'LineWidth', 3);
    
    % Arm 2 (right-back)
    arm2 = [0, 2, 0; arm_length, 2, 0] + pos;
    drone_model.arm2 = plot3(ax, arm2(:,1), arm2(:,2), arm2(:,3), 'Color', edge_color, 'LineWidth', 3);
    
    % Arm 3 (left-front)
    arm3 = [0, -2, 0; -arm_length, -2, 0] + pos;
    drone_model.arm3 = plot3(ax, arm3(:,1), arm3(:,2), arm3(:,3), 'Color', edge_color, 'LineWidth', 3);
    
    % Arm 4 (left-back)
    arm4 = [0, 2, 0; -arm_length, 2, 0] + pos;
    drone_model.arm4 = plot3(ax, arm4(:,1), arm4(:,2), arm4(:,3), 'Color', edge_color, 'LineWidth', 3);
    
    % Create rotors
    theta = linspace(0, 2*pi, 20);
    
    % Rotor 1
    r1_x = arm_length + rotor_radius*cos(theta);
    r1_y = -2 + rotor_radius*sin(theta);
    r1_z = zeros(size(theta));
    drone_model.rotor1 = plot3(ax, r1_x + pos(1), r1_y + pos(2), r1_z + pos(3), 'r-', 'LineWidth', 2);
    
    % Rotor 2
    r2_x = arm_length + rotor_radius*cos(theta);
    r2_y = 2 + rotor_radius*sin(theta);
    r2_z = zeros(size(theta));
    drone_model.rotor2 = plot3(ax, r2_x + pos(1), r2_y + pos(2), r2_z + pos(3), 'r-', 'LineWidth', 2);
    
    % Rotor 3
    r3_x = -arm_length + rotor_radius*cos(theta);
    r3_y = -2 + rotor_radius*sin(theta);
    r3_z = zeros(size(theta));
    drone_model.rotor3 = plot3(ax, r3_x + pos(1), r3_y + pos(2), r3_z + pos(3), 'r-', 'LineWidth', 2);
    
    % Rotor 4
    r4_x = -arm_length + rotor_radius*cos(theta);
    r4_y = 2 + rotor_radius*sin(theta);
    r4_z = zeros(size(theta));
    drone_model.rotor4 = plot3(ax, r4_x + pos(1), r4_y + pos(2), r4_z + pos(3), 'r-', 'LineWidth', 2);
end

function plot_obstacle(ax, center, size)
    % Enhanced obstacle visualization
    half_size = size / 2;
    vertices = [
        center(1)-half_size(1), center(2)-half_size(2), center(3)-half_size(3);
        center(1)+half_size(1), center(2)-half_size(2), center(3)-half_size(3);
        center(1)+half_size(1), center(2)+half_size(2), center(3)-half_size(3);
        center(1)-half_size(1), center(2)+half_size(2), center(3)-half_size(3);
        center(1)-half_size(1), center(2)-half_size(2), center(3)+half_size(3);
        center(1)+half_size(1), center(2)-half_size(2), center(3)+half_size(3);
        center(1)+half_size(1), center(2)+half_size(2), center(3)+half_size(3);
        center(1)-half_size(1), center(2)+half_size(2), center(3)+half_size(3);
    ];
    
    faces = [1 2 3 4; 5 6 7 8; 1 2 6 5; 3 4 8 7; 1 4 8 5; 2 3 7 6];
    patch(ax, 'Faces', faces, 'Vertices', vertices, 'FaceColor', 'red', 'FaceAlpha', 0.3, 'EdgeColor', 'red');
end