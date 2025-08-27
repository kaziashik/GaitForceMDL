% Data
Parkinson = [244.1164, 239.7342, 247.2728, 241.5356];
ALS = [247.9732, 248.0644, 248.3158, 248.5104];
HUNT = [240.985, 244.9942, 243.6466, 245.4948];
Healthy = [248.4621, 245.1254, 246.0226, 244.9984];

% Combine data into a cell array
data = {Parkinson, ALS, HUNT, Healthy};

% Labels
labels = {'Parkinson', 'ALS', 'HUNT', 'Healthy'};

% Create box plot
figure;
boxplot(cell2mat(data'), 'Labels', labels, 'Colors', 'rbgy');
title('Box Plot of Instantaneous RMS for Different Gait');
xlabel('Gait Name');
ylabel('Instantaneous RMS');
grid on;
