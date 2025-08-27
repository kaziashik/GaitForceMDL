% Data
Parkinson_Stdev = [48.7156, 51.9323, 43.5038, 51.9202];
ALS_Stdev = [41.4204, 41.4247, 40.5193, 40.1589];
HUNT_Stdev = [52.9872, 46.6226, 47.998, 45.9013];
Healthy_Stdev = [40.6205, 46.9712, 45.742, 47.3977];

% Combine Stdev data into a cell array
stdev_data = {Parkinson_Stdev, ALS_Stdev, HUNT_Stdev, Healthy_Stdev};

% Labels
labels = {'Parkinson', 'ALS', 'HUNT', 'Healthy'};

% Create box plot
figure;
boxplot(cell2mat(stdev_data'), 'Labels', labels, 'Colors', 'rbgy');
title('Box Plot of Standard Deviation (Stdev) for Different Gait');
xlabel('Gait Name');
ylabel('Standard Deviation (Stdev)');
grid on;
