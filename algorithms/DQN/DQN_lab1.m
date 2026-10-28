clear all; close all; clc;
% Environment Parameters
env_size = 100;
env_height = 50;
drone_start = [10, 10, 10];
target_pos = [80, 80, 30];
target_tolerance = 5;

% Random Obstacles
num_obstacles = 5;
max_size = [15, 15, 18];
fixed_obstacles = zeros(num_obstacles, 6); 
for i = 1:num_obstacles
    x = rand() * env_size;
    y = rand() * env_size;
    z = rand() * env_height;
    sx = 5 + rand() * (max_size(1) - 5);
    sy = 5 + rand() * (max_size(2) - 5);
    sz = 5 + rand() * (max_size(3) - 5);
    fixed_obstacles(i, :) = [x, y, z, sx, sy, sz];
end
fprintf('✓ Using RANDOM obstacles for all episodes\n');

% Actions
step_size = 2.0;
actions = {
    [-step_size, 0, 0], [step_size, 0, 0],
    [0, step_size, 0], [0, -step_size, 0],
    [0, 0, step_size], [0, 0, -step_size]
};
num_actions = 6;

% Network Structure
input_size = 6;
hidden1_size = 64;
hidden2_size = 128;
hidden3_size = 64;
output_size = num_actions;

% Grid search parameters
learning_rates = [0.001, 0.0005];
discount_factors = [0.99, 0.95];
replay_buffer_sizes = [5000,10000];
batch_sizes = [64, 32];
update_target_freqs = [100];
epsilon_starts = [0.9];
epsilon_ends = [0.01];
epsilon_decays = [0.995, 0.999];
num_episodes = 200;
max_steps = 400;
num_runs = 3;
results = struct();
config_idx = 0;

for lr = learning_rates
    for gamma = discount_factors
        for buf_size = replay_buffer_sizes
            for batch_size = batch_sizes
                for tgt_freq = update_target_freqs
                    for eps_start = epsilon_starts
                        for eps_end = epsilon_ends
                            for eps_decay = epsilon_decays
                                config_idx = config_idx + 1;
                                fprintf('\nStarting config %d: lr=%.6f, gamma=%.6f, buf_size=%d, batch_size=%d, tgt_freq=%d, eps_start=%.3f, eps_end=%.3f, eps_decay=%.6f\n', ...
                                    config_idx, lr, gamma, buf_size, batch_size, tgt_freq, eps_start, eps_end, eps_decay);
                                best_avg_reward = -inf;
                                best_model = struct();
                                for run = 1:num_runs
                                    fprintf(' Run %d/%d\n', run, num_runs);
                                    % Initialize weights (He initialization)
                                    W1_online = randn(hidden1_size, input_size) * sqrt(2 / input_size);
                                    b1_online = zeros(hidden1_size, 1);
                                    W2_online = randn(hidden2_size, hidden1_size) * sqrt(2 / hidden1_size);
                                    b2_online = zeros(hidden2_size, 1);
                                    W3_online = randn(hidden3_size, hidden2_size) * sqrt(2 / hidden2_size);
                                    b3_online = zeros(hidden3_size, 1);
                                    W4_online = randn(output_size, hidden3_size) * sqrt(2 / hidden3_size);
                                    b4_online = zeros(output_size, 1);
                                    W1_target = W1_online; b1_target = b1_online;
                                    W2_target = W2_online; b2_target = b2_online;
                                    W3_target = W3_online; b3_target = b3_online;
                                    W4_target = W4_online; b4_target = b4_online;
                                    replay_buffer.states = zeros(buf_size, input_size);
                                    replay_buffer.actions = zeros(buf_size, 1);
                                    replay_buffer.rewards = zeros(buf_size, 1);
                                    replay_buffer.next_states = zeros(buf_size, input_size);
                                    replay_buffer.dones = false(buf_size, 1);
                                    replay_buffer_count = 0;
                                    replay_buffer_pos = 1;
                                    epsilon = eps_start;
                                    epsilon_min = eps_end;
                                    epsilon_decay_val = eps_decay;
                                    episode_rewards = zeros(num_episodes,1);
                                    loss_history = [];
                                    success_rate_history = zeros(num_episodes,1);
                                    epsilon_history = zeros(num_episodes,1);
                                    successes = 0;
                                    failures = 0;
                                    for episode = 1:num_episodes
                                        drone_pos = drone_start;
                                        episode_total_reward = 0;
                                        prev_dist = norm(drone_pos - target_pos);
                                        done = false;
                                        batch_losses = [];
                                        for step = 1:max_steps
                                            state = get_state_vector(drone_pos, target_pos);
                                            [Q_vals, ~, ~, ~] = forward_pass(state, W1_online, b1_online, ...
                                                W2_online, b2_online, ...
                                                W3_online, b3_online, ...
                                                W4_online, b4_online);
                                            if rand() < epsilon
                                                action = randi(num_actions);
                                            else
                                                [~, action] = max(Q_vals);
                                            end
                                            new_pos = drone_pos + actions{action};
                                            reward = 0;
                                            if ~check_bounds(new_pos, env_size, env_height)
                                                reward = -100; done = true;
                                            elseif check_collision(new_pos, fixed_obstacles)
                                                reward = -100; done = true;
                                            else
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
                                                successes = successes + 1;
                                            end
                                            episode_total_reward = episode_total_reward + reward;
                                            prev_dist = dist;
                                            next_state = get_state_vector(drone_pos, target_pos);
                                            replay_buffer.states(replay_buffer_pos, :) = state;
                                            replay_buffer.actions(replay_buffer_pos) = action;
                                            replay_buffer.rewards(replay_buffer_pos) = reward;
                                            replay_buffer.next_states(replay_buffer_pos, :) = next_state;
                                            replay_buffer.dones(replay_buffer_pos) = done;
                                            replay_buffer_pos = mod(replay_buffer_pos, buf_size) + 1;
                                            if replay_buffer_count < buf_size
                                                replay_buffer_count = replay_buffer_count + 1;
                                            end
                                            if replay_buffer_count >= batch_size
                                                indices = randperm(replay_buffer_count, batch_size);
                                                batch_states = replay_buffer.states(indices, :);
                                                batch_actions = replay_buffer.actions(indices);
                                                batch_rewards = replay_buffer.rewards(indices);
                                                batch_next_states = replay_buffer.next_states(indices, :);
                                                batch_dones = replay_buffer.dones(indices);
                                                sum_dW1 = zeros(size(W1_online)); sum_db1 = zeros(size(b1_online));
                                                sum_dW2 = zeros(size(W2_online)); sum_db2 = zeros(size(b2_online));
                                                sum_dW3 = zeros(size(W3_online)); sum_db3 = zeros(size(b3_online));
                                                sum_dW4 = zeros(size(W4_online)); sum_db4 = zeros(size(b4_online));
                                                total_loss = 0;
                                                for i = 1:batch_size
                                                    [current_Q, h1, h2, h3] = forward_pass(batch_states(i,:), ...
                                                        W1_online, b1_online, W2_online, b2_online, ...
                                                        W3_online, b3_online, W4_online, b4_online);
                                                    [next_Q, ~, ~, ~] = forward_pass(batch_next_states(i,:), ...
                                                        W1_target, b1_target, W2_target, b2_target, ...
                                                        W3_target, b3_target, W4_target, b4_target);
                                                    if batch_dones(i)
                                                        target = batch_rewards(i);
                                                    else
                                                        target = batch_rewards(i) + gamma * max(next_Q);
                                                    end
                                                    td_error = target - current_Q(batch_actions(i));
                                                    td_error = max(min(td_error, 5), -5);
                                                    total_loss = total_loss + td_error^2;
                                                    grad_out = zeros(output_size, 1);
                                                    grad_out(batch_actions(i)) = -td_error;
                                                    dW4 = grad_out * h3';
                                                    db4 = grad_out;
                                                    grad_h3 = (W4_online' * grad_out) .* (h3 > 0);
                                                    dW3 = grad_h3 * h2';
                                                    db3 = grad_h3;
                                                    grad_h2 = (W3_online' * grad_h3) .* (h2 > 0);
                                                    dW2 = grad_h2 * h1';
                                                    db2 = grad_h2;
                                                    grad_h1 = (W2_online' * grad_h2) .* (h1 > 0);
                                                    dW1 = grad_h1 * batch_states(i,:);
                                                    db1 = grad_h1;
                                                    sum_dW1 = sum_dW1 + dW1; sum_db1 = sum_db1 + db1;
                                                    sum_dW2 = sum_dW2 + dW2; sum_db2 = sum_db2 + db2;
                                                    sum_dW3 = sum_dW3 + dW3; sum_db3 = sum_db3 + db3;
                                                    sum_dW4 = sum_dW4 + dW4; sum_db4 = sum_db4 + db4;
                                                end
                                                W1_online = W1_online - lr * (sum_dW1 / batch_size);
                                                b1_online = b1_online - lr * (sum_db1 / batch_size);
                                                W2_online = W2_online - lr * (sum_dW2 / batch_size);
                                                b2_online = b2_online - lr * (sum_db2 / batch_size);
                                                W3_online = W3_online - lr * (sum_dW3 / batch_size);
                                                b3_online = b3_online - lr * (sum_db3 / batch_size);
                                                W4_online = W4_online - lr * (sum_dW4 / batch_size);
                                                b4_online = b4_online - lr * (sum_db4 / batch_size);
                                                batch_losses = [batch_losses; total_loss / batch_size];
                                            end
                                            if done
                                                break;
                                            end
                                        end
                                        episode_rewards(episode) = episode_total_reward;
                                        epsilon_history(episode) = epsilon;
                                        if ~done
                                            failures = failures + 1;
                                        end
                                        if mod(episode, tgt_freq) == 0
                                            W1_target = W1_online; b1_target = b1_online;
                                            W2_target = W2_online; b2_target = b2_online;
                                            W3_target = W3_online; b3_target = b3_online;
                                            W4_target = W4_online; b4_target = b4_online;
                                        end
                                        epsilon = max(epsilon_min, epsilon * epsilon_decay_val);
                                        if ~isempty(batch_losses)
                                            loss_history = [loss_history; mean(batch_losses)];
                                        end
                                        total_decided = successes + failures;
                                        success_rate_history(episode) = (successes / max(1, total_decided)) * 100;
                                        % Print progress and plot every 50 episodes
                                        if mod(episode, 50) == 0
                                            fprintf('Ep %d | Success: %d (%.1f%%) | Fail: %d | Avg Reward: %.1f | Loss: %.2f | ε: %.3f\n', ...
                                                episode, successes, success_rate_history(episode), failures, ...
                                                mean(episode_rewards(max(1,episode-49):episode)), mean(loss_history(max(1,end-49):end)), epsilon);
                                            figure(100); clf;
                                            subplot(2,2,1);
                                            plot(1:episode, episode_rewards(1:episode), 'b-', 'LineWidth', 1.5);
                                            hold on;
                                            plot(movmean(episode_rewards(1:episode),10), 'r-', 'LineWidth', 2);
                                            hold off;
                                            xlabel('Episode'); ylabel('Reward');
                                            title('Episode Reward'); grid on;
                                            subplot(2,2,2);
                                            if length(loss_history) >= 10
                                                plot(loss_history, 'b-', 'LineWidth', 1);
                                                hold on;
                                                plot(movmean(loss_history,50), 'r-', 'LineWidth', 2);
                                                hold off;
                                                xlabel('Training Step'); ylabel('Loss');
                                                title('Training Loss'); grid on;
                                            else
                                                text(0.5, 0.5, 'Not enough loss data yet','HorizontalAlignment','center');
                                            end
                                            subplot(2,2,3);
                                            plot(1:episode, success_rate_history(1:episode), 'b-', 'LineWidth', 2);
                                            hold on;
                                            plot(movmean(success_rate_history(1:episode),10), 'r-', 'LineWidth', 2);
                                            hold off;
                                            xlabel('Episode'); ylabel('Success Rate (%)');
                                            title('Success Rate');
                                            ylim([0 100]);
                                            grid on;
                                            subplot(2,2,4);
                                            plot(1:episode, epsilon_history(1:episode), 'm-', 'LineWidth', 2);
                                            xlabel('Episode'); ylabel('Epsilon');
                                            title('Exploration Rate Decay');
                                            ylim([0 1]);
                                            grid on;
                                            drawnow;
                                        end
                                    end
                                    % Save best model if applicable
                                    avg_reward = mean(episode_rewards);
                                    if avg_reward > best_avg_reward
                                        best_avg_reward = avg_reward;
                                        best_model.W1_online = W1_online; best_model.b1_online = b1_online;
                                        best_model.W2_online = W2_online; best_model.b2_online = b2_online;
                                        best_model.W3_online = W3_online; best_model.b3_online = b3_online;
                                        best_model.W4_online = W4_online; best_model.b4_online = b4_online;
                                        best_model.params = struct('lr', lr, 'gamma', gamma, 'buf_size', buf_size, ...
                                            'batch_size', batch_size, 'target_freq', tgt_freq, ...
                                            'epsilon_start', eps_start, 'epsilon_end', eps_end, 'epsilon_decay', eps_decay);
                                        save(sprintf('best_model_config_%d.mat', config_idx), '-struct', 'best_model');
                                        fprintf('✓ Saved best model for config %d with avg reward %.2f\n', config_idx, avg_reward);
                                    end
                                    % Store results for post-analysis
                                    results(config_idx, run).episode_rewards = episode_rewards;
                                    results(config_idx, run).loss_history = loss_history;
                                    results(config_idx, run).success_rate_history = success_rate_history;
                                    results(config_idx, run).epsilon_history = epsilon_history;
                                    results(config_idx, run).params = best_model.params;
                                end % runs
                            end
                        end
                    end
                end
            end
        end
    end
end
fprintf('\nGrid search completed.\n');

% After grid search, plot graphs of best 5 runs by avg reward:
% Gather averages and indices
all_avg_rewards = [];
all_indices = [];
for c = 1:config_idx
    for r = 1:num_runs
        avg_r = mean(results(c,r).episode_rewards);
        all_avg_rewards = [all_avg_rewards; avg_r];
        all_indices = [all_indices; c, r];
    end
end
[~, sorted_idx] = sort(all_avg_rewards, 'descend');
top5_idx = sorted_idx(1:min(5, length(sorted_idx)));
figure('Name','Top 5 Runs Summary','Position',[200 200 1200 600]);
for i = 1:length(top5_idx)
    idx = top5_idx(i);
    c = all_indices(idx,1);
    r = all_indices(idx,2);
    ep_rewards = results(c,r).episode_rewards;
    loss_hist = results(c,r).loss_history;
    success_rate = results(c,r).success_rate_history;
    epsilon_hist = results(c,r).epsilon_history;
    subplot(5,4,(i-1)*4+1);
    plot(ep_rewards, 'b-','LineWidth',1.5);
    hold on; plot(movmean(ep_rewards,10), 'r-', 'LineWidth',2); hold off;
    xlabel('Episode'); ylabel('Reward'); title(sprintf('Run %d-%d Rewards', c, r));
    grid on;
    subplot(5,4,(i-1)*4+2);
    if ~isempty(loss_hist)
        plot(loss_hist, 'b-','LineWidth',1);
        hold on; plot(movmean(loss_hist,50), 'r-', 'LineWidth',2);
        hold off;
        xlabel('Training Update'); ylabel('Loss');
        title(sprintf('Run %d-%d Loss', c, r));
        grid on;
    end
    subplot(5,4,(i-1)*4+3);
    plot(success_rate, 'b-', 'LineWidth',1.5);
    hold on; plot(movmean(success_rate,10), 'r-', 'LineWidth',2);
    hold off;
    xlabel('Episode'); ylabel('Success Rate (%)');
    title(sprintf('Run %d-%d Success Rate', c, r));
    ylim([0 100]); grid on;
    subplot(5,4,(i-1)*4+4);
    plot(epsilon_hist, 'm-', 'LineWidth',1.5);
    xlabel('Episode'); ylabel('Epsilon');
    title(sprintf('Run %d-%d Epsilon Decay', c, r));
    ylim([0 1]); grid on;
end

% Helper functions
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
    in_bounds = all(pos >= [0,0,0]) && all(pos <= [env_size, env_size, env_height]);
end

function [Q_vals, h1, h2, h3] = forward_pass(state, W1, b1, W2, b2, W3, b3, W4, b4)
    h1 = max(0, W1 * state' + b1);
    h2 = max(0, W2 * h1 + b2);
    h3 = max(0, W3 * h2 + b3);
    Q_vals = W4 * h3 + b4;
end
