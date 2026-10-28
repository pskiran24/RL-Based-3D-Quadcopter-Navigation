clear all; close all; clc;
%% 1. ENVIRONMENT SETUP
env_size = 100;
env_height = 50;
drone_start = [10, 10, 10];
target_pos = [80, 80, 30];
target_tolerance = 5;

%% 2. LOAD TRAINED MODEL
try
    load('best_model_config_13.mat'); % Adjust filename appropriately
    disp('Successfully loaded trained model!');
catch
    error('Could not find trained model. Run training and save model first.');
end

%% 3. RANDOM OBSTACLES (same as train)
num_obstacles = 5;
max_size = [15, 15, 18];
generate_random_obstacles = @() ...
    [rand(num_obstacles,1)*env_size, rand(num_obstacles,1)*env_size, ...
    rand(num_obstacles,1)*env_height, ...
    5 + rand(num_obstacles,1)*(max_size(1)-5), ...
    5 + rand(num_obstacles,1)*(max_size(2)-5), ...
    5 + rand(num_obstacles,1)*(max_size(3)-5)];

%% 4. ACTIONS
step_size = 2.0;
actions = {
    [-step_size, 0, 0], [step_size, 0, 0],
    [0, step_size, 0], [0, -step_size, 0],
    [0, 0, step_size], [0, 0, -step_size]
};
num_actions = 6;

%% 5. HELPER FUNCTIONS (Local functions at end of this script)

%% 6. VISUALIZATION SETUP
fig = figure('Name', 'Drone RL Testing with 3D Visualization', 'NumberTitle', 'off', 'Position',[200 100 900 700]);
ax = axes('Parent', fig);
hold(ax, 'on');
axis(ax, 'equal');
grid(ax, 'on');
xlabel(ax, 'X (m)');
ylabel(ax, 'Y (m)');
zlabel(ax, 'Z (m)');
xlim(ax, [0, env_size]);
ylim(ax, [0, env_size]);
zlim(ax, [0, env_height]);
view(ax, 3);
title(ax, 'Drone Navigation Test Episodes');

% Plot target point once
hTarget = plot3(ax, target_pos(1), target_pos(2), target_pos(3), 'g*', 'MarkerSize', 15, 'DisplayName', 'Target');

legend(ax, 'Location', 'northwest');

%% 7. RUN TEST EPISODES
num_test_episodes = 20;
max_steps = 400;
episode_rewards = zeros(num_test_episodes,1);
success_rates = zeros(num_test_episodes,1);
steps_taken = zeros(num_test_episodes,1);
epsilons = zeros(num_test_episodes,1);
success_count = 0;

for episode = 1:num_test_episodes
    obstacles = generate_random_obstacles();
    
    % Clear previous obstacles & drone model
    delete(findobj(ax, 'Type', 'patch', '-and', 'FaceColor', 'red'));
    delete(findobj(ax, 'Type', 'patch', '-and', 'FaceColor', [0.2 0.2 0.8]));
    delete(findobj(ax, 'Type', 'line', '-and', 'Color', 'b'));
    
    % Plot obstacles as translucent cubes
    for i = 1:size(obstacles,1)
        plot_obstacle(ax, obstacles(i,1:3), obstacles(i,4:6));
    end

    drone_pos = drone_start;
    trajectory = zeros(max_steps, 3);
    done = false;
    total_reward = 0;
    prev_dist = norm(drone_pos - target_pos);
    epsilon = 0; % no exploration during evaluation
    
    % Create initial drone model
    drone_model = create_drone_model(ax, drone_pos);
    
    for step = 1:max_steps
        trajectory(step, :) = drone_pos;
        
        state = get_state_vector(drone_pos, target_pos);
        [Q_vals, ~, ~, ~] = forward_pass(state, W1_online, b1_online, W2_online, b2_online, ...
                                         W3_online, b3_online, W4_online, b4_online);
        [~, action] = max(Q_vals);
        new_pos = drone_pos + actions{action};
        
        if ~check_bounds(new_pos, env_size, env_height) || check_collision(new_pos, obstacles)
            reward = -100;
            done = true;
        else
            reward = 0;
            drone_pos = new_pos;
        end
        
        dist = norm(drone_pos - target_pos);
        if ~done
            if dist < prev_dist
                reward = reward + 5;
            else
                reward = reward - 3;
            end
            reward = reward - 0.01 * dist;
        end
        
        if dist < target_tolerance
            reward = reward + 200;
            done = true;
            success_count = success_count + 1;
            success_rates(episode) = 1;
        end
        
        total_reward = total_reward + reward;
        prev_dist = dist;
        
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
        
        drawnow;
        pause(0.05); % Adjust speed of animation here
        
        if done
            steps_taken(episode) = step;
            break;
        end
    end
    
    if ~done
        steps_taken(episode) = max_steps;
        trajectory((step+1):end, :) = NaN;
    else
        trajectory((step+1):end, :) = NaN;
    end
    
    episode_rewards(episode) = total_reward;
    epsilons(episode) = epsilon;

    % Plot final path for this episode
    traj_clean = trajectory(~any(isnan(trajectory), 2), :);
    plot3(ax, traj_clean(:,1), traj_clean(:,2), traj_clean(:,3), 'b-', 'LineWidth', 2, 'DisplayName', 'Drone Path');

    fprintf('Episode %d: Reward=%.1f, Steps=%d, Success=%d\n', episode, total_reward, steps_taken(episode), success_rates(episode));
end

%% 8. PLOTTING KEY METRICS
figure('Name','Drone RL Evaluation Metrics','Position',[200 200 1000 600]);
subplot(3,2,1);
plot(1:num_test_episodes, success_rates*100, 'o-', 'LineWidth', 2);
xlabel('Episode'); ylabel('Success Rate (%)');
title('Success Rate per Episode');
grid on; ylim([0 110]);
subplot(3,2,2);
plot(1:num_test_episodes, episode_rewards, 's-', 'LineWidth', 2);
xlabel('Episode'); ylabel('Episode Reward');
title('Episode Reward');
grid on;
subplot(3,2,3);
plot(1:num_test_episodes, steps_taken, '^-', 'LineWidth', 2);
xlabel('Episode'); ylabel('Steps taken');
title('Episode Length (Steps)');
grid on;

sgtitle('Drone Navigation RL Evaluation Metrics');
fprintf('\nTest Summary:\n');
fprintf('Total Episodes: %d\n', num_test_episodes);
fprintf('Success Rate: %.2f%%\n', (success_count/num_test_episodes)*100);
fprintf('Average Reward: %.2f\n', mean(episode_rewards));
fprintf('Average Steps: %.1f\n', mean(steps_taken));

%% LOCAL HELPER FUNCTIONS
function state = get_state_vector(pos, target_pos)
    dx = target_pos(1) - pos(1);
    dy = target_pos(2) - pos(2);
    dz = target_pos(3) - pos(3);
    state = [pos(1)/100, pos(2)/100, pos(3)/50, dx/100, dy/100, dz/50];
end

function in_collision = check_collision(pos, obstacles)
    in_collision = false;
    for i = 1:size(obstacles, 1)
        if all(abs(pos - obstacles(i,1:3)) <= obstacles(i,4:6)/2)
            in_collision = true;
            return;
        end
    end
end

function in_bounds = check_bounds(pos, env_size, env_height)
    in_bounds = all(pos >= [0, 0, 0]) && all(pos <= [env_size, env_size, env_height]);
end

function [Q_vals, h1, h2, h3] = forward_pass(state, W1, b1, W2, b2, W3, b3, W4, b4)
    h1 = max(0, W1 * state' + b1);
    h2 = max(0, W2 * h1 + b2);
    h3 = max(0, W3 * h2 + b3);
    Q_vals = W4 * h3 + b4;
end

function drone_model = create_drone_model(ax, pos)
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
        'FaceColor', [0.2, 0.2, 0.8], 'EdgeColor', 'blue', 'FaceAlpha', 0.8, 'DisplayName', 'Drone');
    
    arm_length = 5;
    rotor_radius = 1;
    
    arm1 = [0, -2, 0; arm_length, -2, 0] + pos;
    drone_model.arm1 = plot3(ax, arm1(:,1), arm1(:,2), arm1(:,3), 'b-', 'LineWidth', 3);
    
    arm2 = [0, 2, 0; arm_length, 2, 0] + pos;
    drone_model.arm2 = plot3(ax, arm2(:,1), arm2(:,2), arm2(:,3), 'b-', 'LineWidth', 3);
    
    arm3 = [0, -2, 0; -arm_length, -2, 0] + pos;
    drone_model.arm3 = plot3(ax, arm3(:,1), arm3(:,2), arm3(:,3), 'b-', 'LineWidth', 3);
    
    arm4 = [0, 2, 0; -arm_length, 2, 0] + pos;
    drone_model.arm4 = plot3(ax, arm4(:,1), arm4(:,2), arm4(:,3), 'b-', 'LineWidth', 3);
    
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

function plot_obstacle(ax, center, size)
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
    patch(ax, 'Faces', faces, 'Vertices', vertices, 'FaceColor', 'red', 'FaceAlpha', 0.3, 'EdgeColor', 'red', 'DisplayName', 'Obstacle');
end
