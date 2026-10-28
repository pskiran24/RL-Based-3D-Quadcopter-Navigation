clear all; close all; clc;

%% 1. ENVIRONMENT SETUP
env_size = 100;
env_height = 50;
drone_start = [10, 10, 10];
target_pos = [80, 80, 30];
target_tolerance = 5;

%% 2. LOAD TRAINED MODEL
try
    % Load the best SARSA Q-table (adjust filename to match your saved model)
    load('best_SARSA_model_LR0.100_DF0.950_EP0.30.mat', 'Q'); 
    disp('Successfully loaded trained SARSA Q-table!');
    disp(['Q-table dimensions: ', num2str(size(Q))]);
catch
    error('Could not find trained model. Run training script first and save the best Q-table.');
end

%% 3. GRID PARAMETERS (MUST MATCH TRAINING)
grid_resolution = 5;
num_x_bins = ceil(env_size/grid_resolution);
num_y_bins = ceil(env_size/grid_resolution);
num_z_bins = ceil(env_height/grid_resolution);

%% 4. OBSTACLES
num_obstacles = 5;

%% 5. ACTIONS (MUST MATCH TRAINING)
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

%% 6. VISUALIZATION SETUP
fig = figure('Name', 'SARSA Drone Testing - 3D Visualization', 'NumberTitle', 'off', ...
    'Position', [100 100 1200 800]);
ax = axes('Parent', fig);
hold(ax, 'on');
axis(ax, 'equal');
grid(ax, 'on');
xlabel(ax, 'X (m)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel(ax, 'Y (m)', 'FontSize', 12, 'FontWeight', 'bold');
zlabel(ax, 'Z (m)', 'FontSize', 12, 'FontWeight', 'bold');
xlim(ax, [0, env_size]);
ylim(ax, [0, env_size]);
zlim(ax, [0, env_height]);
view(ax, 3);
title(ax, 'SARSA Drone Navigation - Test Episodes', 'FontSize', 14, 'FontWeight', 'bold');

% Plot start position (permanent)
hStart = plot3(ax, drone_start(1), drone_start(2), drone_start(3), 'go', ...
    'MarkerSize', 15, 'MarkerFaceColor', 'g', 'DisplayName', 'Start', 'LineWidth', 2);

% Plot target position with enhanced visibility
hTarget = plot3(ax, target_pos(1), target_pos(2), target_pos(3), 'r^', ...
    'MarkerSize', 20, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k', ...
    'DisplayName', 'Target', 'LineWidth', 3);

% Add a sphere around target for better visibility
[xs, ys, zs] = sphere(20);
surf(ax, xs*3 + target_pos(1), ys*3 + target_pos(2), zs*3 + target_pos(3), ...
    'FaceColor', 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none');

%% 7. RUN TEST EPISODES
num_test_episodes = 100;  % Testing for 100 episodes as requested
max_steps = 200;  % Match training's max_steps_per_episode
episode_rewards = zeros(num_test_episodes, 1);
success_flags = zeros(num_test_episodes, 1);
steps_taken = zeros(num_test_episodes, 1);
success_count = 0;

fprintf('\n========== TESTING SARSA MODEL ==========\n');
fprintf('Running %d test episodes with animated drone...\n\n', num_test_episodes);

for episode = 1:num_test_episodes
    % Generate random obstacles for this episode
    obstacles = generate_random_obstacles(num_obstacles, env_size, env_height);
    
    % Clear previous episode visualizations (obstacles, path, drone)
    delete(findobj(ax, 'Type', 'patch', '-and', 'FaceColor', 'red'));
    delete(findobj(ax, 'Type', 'patch', '-and', 'FaceColor', [0.2 0.2 0.8]));
    delete(findobj(ax, 'Type', 'line', '-and', 'Color', 'b'));
    delete(findobj(ax, 'Type', 'line', '-and', 'Color', 'c'));
    delete(findobj(ax, 'Type', 'line', '-and', 'LineWidth', 3));
    delete(findobj(ax, 'Type', 'line', '-and', 'Color', 'r', '-and', 'LineWidth', 2));
    
    % Plot obstacles
    for i = 1:size(obstacles, 1)
        plot_obstacle(ax, obstacles(i, :));
    end
    
    % Initialize drone
    drone_pos = drone_start;
    trajectory = drone_pos;
    total_reward = 0;
    done = false;
    
    % Greedy policy (no exploration during testing)
    epsilon = 0;
    
    % Create initial drone model
    drone_model = create_drone_model(ax, drone_pos);
    
    for step = 1:max_steps
        % Get current state bins
        [x_bin, y_bin, z_bin] = get_state_bins(drone_pos, grid_resolution, ...
            num_x_bins, num_y_bins, num_z_bins);
        
        % Select action using trained Q-table (greedy)
        action_idx = select_action(Q, x_bin, y_bin, z_bin, epsilon, num_actions);
        
        % Execute action
        new_pos = drone_pos + actions{action_idx};
        
        % Check validity of new position
        if ~check_bounds(new_pos, env_size, env_height) || check_collision(new_pos, obstacles)
            new_pos = drone_pos; % Stay in place if invalid
        end
        
        % Calculate reward (same as training)
        reward = calculate_reward(new_pos, drone_pos, target_pos, obstacles, env_size, env_height);
        total_reward = total_reward + reward;
        
        % Update position
        drone_pos = new_pos;
        trajectory = [trajectory; drone_pos];
        
        % Update drone model visualization
        delete(drone_model.body);
        delete(drone_model.arm1);
        delete(drone_model.arm2);
        delete(drone_model.arm3);
        delete(drone_model.arm4);
        delete(drone_model.rotor1);
        delete(drone_model.rotor2);
        delete(drone_model.rotor3);
        delete(drone_model.rotor4);
        drone_model = create_drone_model(ax, drone_pos);
        
        % Draw trajectory up to current point
        if size(trajectory, 1) > 1
            delete(findobj(ax, 'Type', 'line', '-and', 'Color', 'c'));
            plot3(ax, trajectory(:,1), trajectory(:,2), trajectory(:,3), 'c-', ...
                'LineWidth', 2, 'DisplayName', 'Path');
        end
        
        drawnow;
        pause(0.02); % Animation speed
        
        % Check if target reached
        if norm(drone_pos - target_pos) < target_tolerance
            done = true;
            success_count = success_count + 1;
            success_flags(episode) = 1;
            steps_taken(episode) = step;
            
            % Flash success
            for flash = 1:3
                delete(drone_model.body);
                delete(drone_model.arm1);
                delete(drone_model.arm2);
                delete(drone_model.arm3);
                delete(drone_model.arm4);
                delete(drone_model.rotor1);
                delete(drone_model.rotor2);
                delete(drone_model.rotor3);
                delete(drone_model.rotor4);
                if mod(flash, 2) == 1
                    drone_model = create_drone_model(ax, drone_pos, [0.2 0.8 0.2]); % Green flash
                else
                    drone_model = create_drone_model(ax, drone_pos);
                end
                pause(0.15);
            end
            break;
        end
    end
    
    if ~done
        steps_taken(episode) = max_steps;
    end
    
    episode_rewards(episode) = total_reward;
    
    % Update title with current progress
    title(ax, sprintf('SARSA Test: Episode %d/%d | Success Rate: %.1f%% | Avg Reward: %.1f', ...
        episode, num_test_episodes, (success_count/episode)*100, mean(episode_rewards(1:episode))), ...
        'FontSize', 14, 'FontWeight', 'bold');
    
    % Draw final trajectory for this episode
    delete(findobj(ax, 'Type', 'line', '-and', 'Color', 'c'));
    plot3(ax, trajectory(:,1), trajectory(:,2), trajectory(:,3), 'b-', ...
        'LineWidth', 1.5, 'DisplayName', sprintf('Episode %d', episode));
    
    % Remove drone model before next episode
    pause(0.5);
    delete(drone_model.body);
    delete(drone_model.arm1);
    delete(drone_model.arm2);
    delete(drone_model.arm3);
    delete(drone_model.arm4);
    delete(drone_model.rotor1);
    delete(drone_model.rotor2);
    delete(drone_model.rotor3);
    delete(drone_model.rotor4);
    
    % Print progress
    if mod(episode, 10) == 0
        fprintf('Episode %d/%d: Success Rate = %.1f%%, Avg Reward = %.2f\n', ...
            episode, num_test_episodes, (success_count/episode)*100, mean(episode_rewards(1:episode)));
    end
end

%% 8. FINAL STATISTICS
fprintf('\n========== TEST RESULTS ==========\n');
fprintf('Total Episodes:       %d\n', num_test_episodes);
fprintf('Successful Episodes:  %d\n', success_count);
fprintf('Success Rate:         %.2f%%\n', (success_count/num_test_episodes)*100);
fprintf('Average Reward:       %.2f\n', mean(episode_rewards));
fprintf('Std Reward:           %.2f\n', std(episode_rewards));
fprintf('Average Steps:        %.1f\n', mean(steps_taken));
fprintf('Min Reward:           %.2f\n', min(episode_rewards));
fprintf('Max Reward:           %.2f\n', max(episode_rewards));
fprintf('==================================\n\n');

%% 9. PLOTTING KEY METRICS
figure('Name', 'SARSA Test Evaluation Metrics', 'Position', [100, 100, 1400, 800]);

% Success rate per episode
subplot(2, 3, 1);
plot(1:num_test_episodes, cumsum(success_flags)./(1:num_test_episodes)', 'b-', 'LineWidth', 2);
xlabel('Episode', 'FontSize', 11);
ylabel('Cumulative Success Rate', 'FontSize', 11);
title('Success Rate Over Episodes', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
ylim([0 1]);
yline((success_count/num_test_episodes), 'r--', 'LineWidth', 2, ...
    'DisplayName', sprintf('Final: %.1f%%', (success_count/num_test_episodes)*100));
legend('Location', 'best');

% Episode rewards
subplot(2, 3, 2);
plot(1:num_test_episodes, episode_rewards, 'o-', 'LineWidth', 1.5, 'MarkerSize', 4);
hold on;
plot(1:num_test_episodes, movmean(episode_rewards, 10), 'r-', 'LineWidth', 2.5);
xlabel('Episode', 'FontSize', 11);
ylabel('Total Reward', 'FontSize', 11);
title('Episode Rewards', 'FontSize', 12, 'FontWeight', 'bold');
legend('Raw', 'Moving Avg (10)', 'Location', 'best');
grid on;

% Steps taken
subplot(2, 3, 3);
bar(steps_taken, 'FaceColor', [0.3 0.6 0.8]);
xlabel('Episode', 'FontSize', 11);
ylabel('Steps Taken', 'FontSize', 11);
title('Episode Length (Steps)', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
yline(mean(steps_taken), 'r--', 'LineWidth', 2, 'DisplayName', 'Average');
legend('Location', 'best');

% Reward distribution
subplot(2, 3, 4);
histogram(episode_rewards, 20, 'FaceColor', [0.2 0.7 0.4], 'EdgeColor', 'k');
xlabel('Total Reward', 'FontSize', 11);
ylabel('Frequency', 'FontSize', 11);
title('Reward Distribution', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
xline(mean(episode_rewards), 'r--', 'LineWidth', 2, 'DisplayName', 'Mean');
legend('Location', 'best');

% Success vs Failure comparison
subplot(2, 3, 5);
success_rewards = episode_rewards(success_flags == 1);
failure_rewards = episode_rewards(success_flags == 0);
if ~isempty(success_rewards) && ~isempty(failure_rewards)
    boxplot([success_rewards; failure_rewards], [ones(length(success_rewards),1); 2*ones(length(failure_rewards),1)], ...
        'Labels', {'Success', 'Failure'}, 'Colors', [0.2 0.8 0.3; 0.8 0.2 0.2]);
    ylabel('Total Reward', 'FontSize', 11);
    title('Success vs Failure Rewards', 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
elseif ~isempty(success_rewards)
    bar([mean(success_rewards), 0], 'FaceColor', [0.2 0.8 0.3]);
    set(gca, 'XTickLabel', {'Success', 'Failure'});
    ylabel('Total Reward', 'FontSize', 11);
    title('Success vs Failure Rewards', 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
else
    bar([0, mean(failure_rewards)], 'FaceColor', [0.8 0.2 0.2]);
    set(gca, 'XTickLabel', {'Success', 'Failure'});
    ylabel('Total Reward', 'FontSize', 11);
    title('Success vs Failure Rewards', 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
end

% Steps for successful episodes
subplot(2, 3, 6);
if success_count > 0
    success_steps = steps_taken(success_flags == 1);
    histogram(success_steps, 15, 'FaceColor', [0.2 0.8 0.3], 'EdgeColor', 'k');
    xlabel('Steps', 'FontSize', 11);
    ylabel('Frequency', 'FontSize', 11);
    title(sprintf('Steps in Successful Episodes (n=%d)', success_count), ...
        'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    xline(mean(success_steps), 'r--', 'LineWidth', 2, 'DisplayName', 'Mean');
    legend('Location', 'best');
else
    text(0.5, 0.5, 'No Successful Episodes', 'HorizontalAlignment', 'center', ...
        'FontSize', 14, 'Color', 'red', 'FontWeight', 'bold');
    axis off;
end

sgtitle(sprintf('SARSA Model Testing Results (100 Episodes) - Success Rate: %.1f%%', ...
    (success_count/num_test_episodes)*100), 'FontSize', 16, 'FontWeight', 'bold');

%% 10. HELPER FUNCTIONS

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
    % Distance-based reward
    old_dist = norm(old_pos - target_pos);
    new_dist = norm(new_pos - target_pos);
    
    if new_dist < old_dist
        reward = 1;  % Moving closer
    else
        reward = -0.5;  % Moving away
    end
    
    % Collision penalty
    if check_collision(new_pos, obstacles)
        reward = reward - 100;
    end
    
    % Out of bounds penalty
    if ~check_bounds(new_pos, env_size, env_height)
        reward = reward - 100;
    end
    
    % Success bonus
    if new_dist < 5
        reward = reward + 100;
    end
end

function plot_obstacle(ax, obs)
    % obs = [x, y, z, size_x, size_y, size_z]
    center = obs(1:3);
    sizes = obs(4:6);
    half_size = sizes / 2;
    
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
    
    patch(ax, 'Faces', faces, 'Vertices', vertices, ...
        'FaceColor', 'red', 'FaceAlpha', 0.3, 'EdgeColor', 'red', ...
        'LineWidth', 1, 'DisplayName', 'Obstacle');
end

function drone_model = create_drone_model(ax, pos, body_color)
    % Optional body color parameter (default blue)
    if nargin < 3
        body_color = [0.2, 0.2, 0.8];
    end
    
    body_size = 1.5;
    
    % Create body vertices (cuboid)
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
    
    % Draw drone body
    drone_model.body = patch(ax, 'Faces', body_faces, 'Vertices', body_vertices, ...
        'FaceColor', body_color, 'EdgeColor', 'blue', 'FaceAlpha', 0.8, 'DisplayName', 'Drone');
    
    arm_length = 5;
    rotor_radius = 1;
    
    % Draw four arms
    arm1 = [0, -2, 0; arm_length, -2, 0] + pos;
    drone_model.arm1 = plot3(ax, arm1(:,1), arm1(:,2), arm1(:,3), 'b-', 'LineWidth', 3);
    
    arm2 = [0, 2, 0; arm_length, 2, 0] + pos;
    drone_model.arm2 = plot3(ax, arm2(:,1), arm2(:,2), arm2(:,3), 'b-', 'LineWidth', 3);
    
    arm3 = [0, -2, 0; -arm_length, -2, 0] + pos;
    drone_model.arm3 = plot3(ax, arm3(:,1), arm3(:,2), arm3(:,3), 'b-', 'LineWidth', 3);
    
    arm4 = [0, 2, 0; -arm_length, 2, 0] + pos;
    drone_model.arm4 = plot3(ax, arm4(:,1), arm4(:,2), arm4(:,3), 'b-', 'LineWidth', 3);
    
    % Draw rotors (circles at arm ends)
    theta = linspace(0, 2*pi, 20);
    
    % Rotor 1 (right front)
    r1_x = arm_length + rotor_radius*cos(theta);
    r1_y = -2 + rotor_radius*sin(theta);
    r1_z = zeros(size(theta));
    drone_model.rotor1 = plot3(ax, r1_x + pos(1), r1_y + pos(2), r1_z + pos(3), 'r-', 'LineWidth', 2);
    
    % Rotor 2 (right back)
    r2_x = arm_length + rotor_radius*cos(theta);
    r2_y = 2 + rotor_radius*sin(theta);
    r2_z = zeros(size(theta));
    drone_model.rotor2 = plot3(ax, r2_x + pos(1), r2_y + pos(2), r2_z + pos(3), 'r-', 'LineWidth', 2);
    
    % Rotor 3 (left front)
    r3_x = -arm_length + rotor_radius*cos(theta);
    r3_y = -2 + rotor_radius*sin(theta);
    r3_z = zeros(size(theta));
    drone_model.rotor3 = plot3(ax, r3_x + pos(1), r3_y + pos(2), r3_z + pos(3), 'r-', 'LineWidth', 2);
    
    % Rotor 4 (left back)
    r4_x = -arm_length + rotor_radius*cos(theta);
    r4_y = 2 + rotor_radius*sin(theta);
    r4_z = zeros(size(theta));
    drone_model.rotor4 = plot3(ax, r4_x + pos(1), r4_y + pos(2), r4_z + pos(3), 'r-', 'LineWidth', 2);
end