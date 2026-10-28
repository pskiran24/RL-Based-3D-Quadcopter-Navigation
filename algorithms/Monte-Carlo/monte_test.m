clear all; close all; clc;
%% 1. ENVIRONMENT SETUP
env_size = 100;
env_height = 50;
drone_start = [10, 10, 10];
target_pos = [80, 80, 30];
target_tolerance = 5;
%% 2. LOAD TRAINED MONTE CARLO MODEL
try
    load('best_MC_model_DF0.950_EP0.20.mat', 'Q', 'N');
    disp('Successfully loaded trained Monte Carlo Q-table!');
    disp(['Q-table dimensions: ', num2str(size(Q))]);
catch
    error('Could not find trained Monte Carlo model. Run training script first and save Q and N.');
end
%% 3. GRID PARAMETERS
grid_resolution = 5;
num_x_bins = ceil(env_size/grid_resolution);
num_y_bins = ceil(env_size/grid_resolution);
num_z_bins = ceil(env_height/grid_resolution);
%% 4. OBSTACLES
num_obstacles = 5;
%% 5. ACTIONS
step_size = 2;
actions = {
    [-step_size, 0, 0],
    [step_size, 0, 0],
    [0, step_size, 0],
    [0, -step_size, 0],
    [0, 0, step_size],
    [0, 0, -step_size]
};
num_actions = length(actions);
%% 6. VISUALIZATION SETUP
fig = figure('Name', 'MONTE CARLO Drone Testing - 3D Visualization', 'NumberTitle', 'off', ...
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
title(ax, 'MONTE CARLO Drone Navigation - Test Episodes', 'FontSize', 14, 'FontWeight', 'bold');
hStart = plot3(ax, drone_start(1), drone_start(2), drone_start(3), 'go', ...
    'MarkerSize', 15, 'MarkerFaceColor', 'g', 'DisplayName', 'Start', 'LineWidth', 2);
hTarget = plot3(ax, target_pos(1), target_pos(2), target_pos(3), 'r^', ...
    'MarkerSize', 20, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k', ...
    'DisplayName', 'Target', 'LineWidth', 3);
[xs, ys, zs] = sphere(20);
surf(ax, xs*3 + target_pos(1), ys*3 + target_pos(2), zs*3 + target_pos(3), ...
    'FaceColor', 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
%% 7. RUN TEST EPISODES
num_test_episodes = 100;
max_steps = 200;
episode_rewards = zeros(num_test_episodes, 1);
success_flags = zeros(num_test_episodes, 1);
steps_taken = zeros(num_test_episodes, 1);
success_count = 0;
fprintf('\n========== TESTING MONTE CARLO MODEL ==========\n');
fprintf('Running %d test episodes with animated drone...\n\n', num_test_episodes);
for episode = 1:num_test_episodes
    obstacles = generate_random_obstacles(num_obstacles, env_size, env_height);
    
    delete(findobj(ax, 'Type', 'patch', '-and', 'FaceColor', 'red'));
    delete(findobj(ax, 'Type', 'patch', '-and', 'FaceColor', [0.2 0.2 0.8]));
    delete(findobj(ax, 'Type', 'line', '-and', 'Color', 'b'));
    delete(findobj(ax, 'Type', 'line', '-and', 'Color', 'c'));
    delete(findobj(ax, 'Type', 'line', '-and', 'LineWidth', 3));
    delete(findobj(ax, 'Type', 'line', '-and', 'Color', 'r', '-and', 'LineWidth', 2));
    
    for i = 1:size(obstacles, 1)
        plot_obstacle(ax, obstacles(i, :));
    end
    
    drone_pos = drone_start;
    trajectory = drone_pos;
    total_reward = 0;
    epsilon = 0; % greedy
    
    drone_model = create_drone_model(ax, drone_pos);
    
    for step = 1:max_steps
        [x_bin, y_bin, z_bin] = get_state_bins(drone_pos, grid_resolution, ...
            num_x_bins, num_y_bins, num_z_bins);
        
        if rand() < epsilon
            action_idx = randi(num_actions);
        else
            [~, action_idx] = max(Q(x_bin, y_bin, z_bin, :));
        end
        
        new_pos = drone_pos + actions{action_idx};
        if ~check_bounds(new_pos, env_size, env_height) || check_collision(new_pos, obstacles)
            new_pos = drone_pos;
        end
        
        reward = calculate_reward(new_pos, drone_pos, target_pos, obstacles, env_size, env_height);
        total_reward = total_reward + reward;
        
        drone_pos = new_pos;
        trajectory = [trajectory; drone_pos];
        
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
        
        if size(trajectory,1) > 1
            delete(findobj(ax, 'Type', 'line', '-and', 'Color', 'c'));
            plot3(ax, trajectory(:,1), trajectory(:,2), trajectory(:,3), 'c-', 'LineWidth', 2);
        end
        
        drawnow;
        pause(0.02);
        
        if norm(drone_pos - target_pos) < target_tolerance
            success_count = success_count + 1;
            success_flags(episode) = 1;
            steps_taken(episode) = step;
            for flash=1:3
                delete(drone_model.body);
                delete(drone_model.arm1);
                delete(drone_model.arm2);
                delete(drone_model.arm3);
                delete(drone_model.arm4);
                delete(drone_model.rotor1);
                delete(drone_model.rotor2);
                delete(drone_model.rotor3);
                delete(drone_model.rotor4);
                if mod(flash,2)==1
                    drone_model=create_drone_model(ax,drone_pos,[0.2 0.8 0.2]);
                else
                    drone_model=create_drone_model(ax,drone_pos);
                end
                pause(0.15);
            end
            break;
        end
    end
    
    if norm(drone_pos - target_pos) >= target_tolerance
        steps_taken(episode) = max_steps;
    end
    
    episode_rewards(episode) = total_reward;
    
    title(ax, sprintf('MONTE CARLO Test: Episode %d/%d | Success Rate: %.1f%% | Avg Reward: %.1f', ...
        episode, num_test_episodes, (success_count/episode)*100, mean(episode_rewards(1:episode))), ...
        'FontSize', 14, 'FontWeight', 'bold');
    
    delete(findobj(ax,'Type','line','-and','Color','c'));
    plot3(ax, trajectory(:,1),trajectory(:,2),trajectory(:,3),'b-', 'LineWidth', 1.5);
    
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
    
    if mod(episode,10) == 0
        fprintf('Episode %d/%d: Success Rate = %.1f%%, Avg Reward = %.2f\n', ...
            episode, num_test_episodes, (success_count/episode)*100, mean(episode_rewards(1:episode)));
    end
end

% Helper functions defined once below:

function obstacles = generate_random_obstacles(num_obs, env_size, env_height)
    obstacles = zeros(num_obs, 6);
    for i = 1:num_obs
        x = rand() * (env_size - 20) + 10;
        y = rand() * (env_size - 20) + 10;
        z = rand() * (env_height - 20) + 10;
        size_val = rand() * 5 + 10;
        obstacles(i,:) = [x, y, z, size_val, size_val, size_val];
    end
end

function [x_bin, y_bin, z_bin] = get_state_bins(pos, grid_resolution, num_x_bins, num_y_bins, num_z_bins)
    x_bin = min(max(ceil(pos(1)/grid_resolution),1), num_x_bins);
    y_bin = min(max(ceil(pos(2)/grid_resolution),1), num_y_bins);
    z_bin = min(max(ceil(pos(3)/grid_resolution),1), num_z_bins);
end

function collision = check_collision(pos, obstacles)
    collision = false;
    for i=1:size(obstacles,1)
        obs_center = obstacles(i,1:3);
        obs_size = obstacles(i,4:6);
        if all(abs(pos - obs_center) <= obs_size/2)
            collision = true;
            return;
        end
    end
end

function valid = check_bounds(pos, env_size, env_height)
    valid = pos(1) >=0 && pos(1) <= env_size && ...
        pos(2) >=0 && pos(2) <= env_size && ...
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

function plot_obstacle(ax, obs)
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
    faces = [1 2 3 4;5 6 7 8;1 2 6 5;3 4 8 7;1 4 8 5;2 3 7 6];
    patch(ax,'Faces',faces,'Vertices',vertices,'FaceColor','red',...
        'FaceAlpha',0.3,'EdgeColor','red','LineWidth',1);
end

function drone_model = create_drone_model(ax, pos, body_color)
    if nargin < 3
        body_color = [0.2, 0.2, 0.8];
    end
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
    body_faces = [1 2 3 4;5 6 7 8;1 2 6 5;3 4 8 7;1 4 8 5;2 3 7 6];
    drone_model.body = patch(ax,'Faces',body_faces,'Vertices',body_vertices,...
        'FaceColor',body_color,'EdgeColor','blue','FaceAlpha',0.8);
    
    arm_length = 5;
    rotor_radius = 1;
    arms = [
        0 -2 0; arm_length -2 0;
        0 2 0; arm_length 2 0;
        0 -2 0; -arm_length -2 0;
        0 2 0; -arm_length 2 0
    ];
    drone_model.arm1 = plot3(ax, arms(1:2,1)+pos(1), arms(1:2,2)+pos(2), arms(1:2,3)+pos(3), 'b-', 'LineWidth', 3);
    drone_model.arm2 = plot3(ax, arms(3:4,1)+pos(1), arms(3:4,2)+pos(2), arms(3:4,3)+pos(3), 'b-', 'LineWidth', 3);
    drone_model.arm3 = plot3(ax, arms(5:6,1)+pos(1), arms(5:6,2)+pos(2), arms(5:6,3)+pos(3), 'b-', 'LineWidth', 3);
    drone_model.arm4 = plot3(ax, arms(7:8,1)+pos(1), arms(7:8,2)+pos(2), arms(7:8,3)+pos(3), 'b-', 'LineWidth', 3);
    
    theta = linspace(0, 2*pi, 20);
    r1_x = arm_length + rotor_radius*cos(theta);
    r1_y = -2 + rotor_radius*sin(theta);
    r1_z = zeros(size(theta));
    drone_model.rotor1 = plot3(ax, r1_x + pos(1), r1_y + pos(2), r1_z + pos(3), 'r-', 'LineWidth', 2);
    
    r2_x = arm_length + rotor_radius*cos(theta);
    r2_y = 2 + rotor_radius*sin(theta);
    r2_z = zeros(size(theta));
    drone_model.rotor2 = plot3(ax, r2_x + pos(1), r2_y + pos(2), r2_z + pos(3), 'r-', 'LineWidth', 2);
    
    r3_x = -arm_length + rotor_radius*cos(theta);
    r3_y = -2 + rotor_radius*sin(theta);
    r3_z = zeros(size(theta));
    drone_model.rotor3 = plot3(ax, r3_x + pos(1), r3_y + pos(2), r3_z + pos(3), 'r-', 'LineWidth', 2);
    
    r4_x = -arm_length + rotor_radius*cos(theta);
    r4_y = 2 + rotor_radius*sin(theta);
    r4_z = zeros(size(theta));
    drone_model.rotor4 = plot3(ax, r4_x + pos(1), r4_y + pos(2), r4_z + pos(3), 'r-', 'LineWidth', 2);
end
