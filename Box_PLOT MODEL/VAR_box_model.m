% VAR data
Parkinson_VAR = [2373.2053, 2696.9625, 1892.5775, 2695.7096];
ALS_VAR = [1715.6534, 1716.0046, 1641.8152, 1612.734];
HUNT_VAR = [2807.6482, 2173.6687, 2303.8779, 2106.932];
Healthy_VAR = [1650.0275, 2206.291, 2092.3263, 2246.5384];

% Combine VAR data into a cell array
VAR_data = {Parkinson_VAR, ALS_VAR, HUNT_VAR, Healthy_VAR};

% Labels
labels = {'Parkinson', 'ALS', 'HUNT', 'Healthy'};

% Create box plot
figure;
boxplot(cell2mat(VAR_data'), 'Labels', labels, 'Colors', 'rbgy');
title('Box Plot of Variance (VAR) for Different Gait');
xlabel('Gait Name');
ylabel('Variance (VAR)');
grid on;
