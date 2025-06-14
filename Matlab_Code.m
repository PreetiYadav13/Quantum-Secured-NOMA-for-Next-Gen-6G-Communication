% Enhanced MATLAB Code for Multi-User NOMA with Q-learning and QKD + Rayleigh Fading
clear; clc; close all;

%% Parameters
numUsers = 2 ;
modulationOrders = [4, 16, 64];
SNRdB = 0:2:30;
SNR = 10.^(SNRdB/10);
totalPower = 1;
numSymbols = 1e6;

%% User Distances and Rayleigh Fading Signal Strength
userDistances = randi([80, 130], 1, numUsers); 
pathLossExponent = 3;
pathLoss = 1 ./ (userDistances.^pathLossExponent);

% Rayleigh fading component (mean power = 1)
rayleighFading = abs(sqrt(0.5) * (randn(1, numUsers) + 1i * randn(1, numUsers)));

% Combine path loss and fading
signalStrength = pathLoss .* rayleighFading;

% Keep a normalized version for plotting
signalStrengthNorm = signalStrength / max(signalStrength);

% Sort users by true signal strength
[~, userRank] = sort(signalStrength);

%% BER Calculation
BER = zeros(length(modulationOrders), length(SNRdB));
for m = 1:length(modulationOrders)
    M = modulationOrders(m);
    k = log2(M);
    for s = 1:length(SNRdB)
        EbN0_dB = SNRdB(s) - 10*log10(k);
        BER(m, s) = berawgn(EbN0_dB, 'qam', M);
    end
end
BER = max(BER, 1e-9);
BER = smoothdata(BER, 'movmean', 3);

%% QBER Calculation
QBER = zeros(length(modulationOrders), length(SNR));
for m = 1:length(modulationOrders)
    decayFactor = 4 + (m-1)*2;
    QBER(m,:) = 0.5 * exp(-SNR / decayFactor);
end

%% Secret Key Rate (SKR)
SKR = zeros(length(modulationOrders), length(SNR));
rawRate = 1e6;
eta = 0.1;
p_dark = 1e-5;
beta = 0.95;
delta_m = 0.02;
h = @(x) -x .* log2(x + eps) - (1 - x) .* log2(1 - x + eps);
for m = 1:length(modulationOrders)
    for s = 1:length(SNR)
        qber = QBER(m, s);
        detectionProb = eta * SNR(s);
        I_AB = 1 - h(qber);
        chi_BE = h(qber);
        rate = beta * I_AB - chi_BE - delta_m;
        rate = max(rate, 0);
        SKR(m, s) = rawRate * detectionProb * rate;
    end
end

%% Q-Learning Power Allocation
numEpisodes = 5000;
nActions = 20;
Q_table = zeros(numUsers, nActions);
reward_history = zeros(numUsers, numEpisodes);
powerAlloc_history = zeros(numUsers, numEpisodes);
alpha = 0.1; gamma = 0.95;
initialEpsilon = 0.5; finalEpsilon = 0.05;
epsilonDecay = (initialEpsilon - finalEpsilon) / numEpisodes;
powerLevels = linspace(0.01, 0.99, nActions);

for ep = 1:numEpisodes
    epsilon = max(finalEpsilon, initialEpsilon - epsilonDecay * ep);
    snrIdx = min(length(SNR), ceil(ep / (numEpisodes/length(SNR))));
    currentSNR = SNR(snrIdx);

    % --- Joint Action Selection for 2 Users ---
    if rand < epsilon
        action1 = randi(nActions);
        action2 = randi(nActions);
    else
        [~, action1] = max(Q_table(1, :));
        [~, action2] = max(Q_table(2, :));
    end

    % Raw allocation
    allocPowers = [powerLevels(action1); powerLevels(action2)];

    % Normalize total power
    allocPowers = allocPowers / sum(allocPowers) * totalPower;

    % Enforce that weaker user gets more power
    [~, sortedIdx] = sort(signalStrength, 'ascend');
    allocPowers(sortedIdx) = sort(allocPowers(sortedIdx), 'descend');

    powerAlloc_history(:, ep) = allocPowers;

    for user = 1:numUsers
        channelGain = signalStrength(user);
        SINR_val = allocPowers(user) * channelGain * currentSNR;
        reward = log2(1 + SINR_val);

        % Bias reward: amplify reward for weak user power
        strengthRank = find(user == sortedIdx); % 1 = weakest
        bias = (numUsers - strengthRank + 1) / numUsers; % 1 for weakest
        reward = reward * bias^2;

        % Extra penalty if strong user gets more power
        if strengthRank > 1 && allocPowers(user) > allocPowers(sortedIdx(1))
            reward = reward * 0.3;
        end

        % Normalize reward
        reward = reward / 5; % Scale
        reward = max(min(reward, 1), 0); % Clip

        % Q-Update using action before re-sorting
        if user == 1
            action = action1;
        else
            action = action2;
        end

        Q_table(user, action) = (1 - alpha) * Q_table(user, action) + ...
            alpha * (reward + gamma * max(Q_table(user, :)));

        reward_history(user, ep) = reward;
    end
end

powerAlloc = mean(powerAlloc_history(:, end-100:end), 2);

%% Final SINR
SINR_final = (powerAlloc .* signalStrength) .* SNR(end);
avgSumRate = sum(log2(1 + SINR_final), 1);

%% Fairness Index
numerator = (sum(powerAlloc))^2;
denominator = numUsers * sum(powerAlloc.^2);
fairnessIndex = numerator / denominator;


%% Average Sum Rate vs SNR for All QAM Orders
avgSumRate_vs_SNR_all = zeros(length(modulationOrders), length(SNR));

for m = 1:length(modulationOrders)
    M = modulationOrders(m);
    k = log2(M); % Bits per symbol
    for s = 1:length(SNR)
        currentSNR = SNR(s);
        
        % Adjust SINR based on QAM modulation
        SINR_users = (powerAlloc(:) .* signalStrength(:)) * currentSNR;
        SINR_eff = SINR_users / k; % Normalize for higher-order modulation

        rates = log2(1 + SINR_eff); % Rate per user
        avgSumRate_vs_SNR_all(m, s) = sum(rates); % Total system sum rate
    end
end



%% PLOTS

% BER
figure;
semilogy(SNRdB, BER(1,:), 'o-', SNRdB, BER(2,:), 's-', SNRdB, BER(3,:), 'd-');
xlabel('SNR (dB)'); ylabel('BER'); legend('4-QAM', '16-QAM', '64-QAM');
title('BER vs SNR'); grid on;

% QBER
figure;
plot(SNRdB, QBER(1,:), 'o-', SNRdB, QBER(2,:), 's-', SNRdB, QBER(3,:), 'd-');
xlabel('SNR (dB)'); ylabel('QBER'); legend('4-QAM', '16-QAM', '64-QAM');
title('QBER vs SNR'); grid on;

% SKR
figure;
plot(SNRdB, SKR(1,:), 'o-', SNRdB, SKR(2,:), 's-', SNRdB, SKR(3,:), 'd-');
xlabel('SNR (dB)'); ylabel('Secret Key Rate (bps)');
legend('4-QAM', '16-QAM', '64-QAM');
title('Secret Key Rate vs SNR'); grid on;

% Q-Learning Cumulative Reward
figure;
hold on;
cumReward = cumsum(reward_history, 2);
for user = 1:numUsers
    plot(cumReward(user,:), 'LineWidth', 1.5, 'DisplayName', sprintf('User %d', user));
end
xlabel('Episode'); ylabel('Cumulative Reward');
title('Q-Learning Cumulative Reward vs Episode');
legend show; grid on;

% Q-Learning Average Reward
figure;
hold on;
avgReward = movmean(reward_history, 100, 2);
for user = 1:numUsers
    plot(avgReward(user,:), 'LineWidth', 1.5, 'DisplayName', sprintf('User %d', user));
end
xlabel('Episode'); ylabel('Average Reward (Moving Avg)');
title('Q-Learning Average Reward vs Episode');
legend show; grid on;

% Normalized Signal Strength for visualization
figure;
bar(categorical(arrayfun(@(x) sprintf('User %d', x), 1:numUsers, 'UniformOutput', false)), signalStrengthNorm);
xlabel('User'); ylabel('Normalized Signal Strength');
title('User Signal Strength Comparison');

% Q-Table Heatmap
figure;
for user = 1:numUsers
    subplot(1, numUsers, user);
    imagesc(powerLevels, 1, Q_table(user, :));
    colorbar;
    xlabel('Power Level'); ylabel('Q-Value');
    title(sprintf('Q-Table - User %d', user));
end

% Power Allocation - Stacked Bar Chart
figure;
bar(1, powerAlloc', 'stacked');
xticks(1);
xticklabels({'Total Power'});
ylabel('Power Allocation');
legend(arrayfun(@(x) sprintf('User %d', x), 1:numUsers, 'UniformOutput', false), 'Location', 'northeastoutside');
title('Final Power Allocation - Stacked Bar Chart');
ylim([0 1]);


% Fairness Index and Power Allocation
disp(['Jain''s Fairness Index: ', num2str(fairnessIndex)]);
for i = 1:numUsers
    disp(['Power Allocation: User ', num2str(i), ' = ', num2str(powerAlloc(i))]);
end

%% Plot Average Sum Rate vs SNR for 3 QAMs
figure;
plot(SNRdB, avgSumRate_vs_SNR_all(1,:), 'o-', 'LineWidth', 1.5); hold on;
plot(SNRdB, avgSumRate_vs_SNR_all(2,:), 's--', 'LineWidth', 1.5);
plot(SNRdB, avgSumRate_vs_SNR_all(3,:), 'd-.', 'LineWidth', 1.5);
xlabel('SNR (dB)');
ylabel('Average Sum Rate (bps/Hz)');
title('Average Sum Rate vs SNR for Different QAM Orders');
legend('4-QAM', '16-QAM', '64-QAM', 'Location', 'northwest');
grid on;
