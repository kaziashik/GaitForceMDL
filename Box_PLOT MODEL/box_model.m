% Data
Parkinson = [239.2062, 234.0417, 243.4158, 235.8892];
ALS = [244.4894, 244.5811, 244.9876, 245.2442];
HUNT = [235.0875, 240.5171, 238.8719, 241.1654];
Healthy = [245.1192, 240.583, 241.7329, 240.3699];

% Combine data into a cell array
data = {Parkinson, ALS, HUNT, Healthy};

% Labels
labels = {'Parkinson', 'ALS', 'HUNT', 'Healthy'};

% Create box plot
figure;
boxplot(cell2mat(data'), 'Labels', labels, 'Colors', 'rbgy');
title('Box Plot of Mean Values for Different Gait');
xlabel('Gait Name');
ylabel('Mean Value');
grid on;