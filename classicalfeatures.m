function features = classicalfeatures(image)
    % Convert the image to grayscale if it's in RGB
    if size(image, 3) == 3
        image = rgb2gray(image);
    end
    
    % Feature extraction
    mean_intensity = mean(image(:));
    std_dev_intensity = std(double(image(:)));
    range_intensity = range(image(:));
    
    % Additional features
    median_intensity = median(image(:));
    interquartile_range = iqr(image(:));

    % ModeROI
    counts = imhist(image);
    [~, idx] = max(counts);
    mode_intensity = (idx - 1) / 255; % Convert to range [0, 1]
    
    % StdMode
    % std_mode = std(double(image(image == mode_intensity)));

    % MeanBound
    mean_bound = mean(image(:)) - min(image(:));

    % Area, Perimeter, Compactness
    stats = regionprops(image > graythresh(image), 'Area', 'Perimeter');
    area = stats.Area;
    perimeter = stats.Perimeter;
    compactness = 4 * pi * area / perimeter^2;

    % Skewness and Kurtosis
    % skewness_val = skewness(image(:));
    % kurtosis_val = kurtosis(image(:));

    % MGR and VGR
    mgr = sum(abs(diff(image(:)))) / numel(image);
    vgr = sum(abs(diff(diff(image(:))))) / numel(image);

    % Root Mean Square (RMS)
    rms_val = sqrt(mean(image(:).^2));

    % Integrated EMG (IEMG)
    iemg = sum(abs(image(:)));

    % Simple Square Integral (SSI)
    ssi = sum(image(:).^2);

    % Variance (VAR)
    var_val = var(double(image(:)));

    % Area of Power
    area_power = sum((abs(fft(image(:)))/length(image)).^2);

    % Spectral Moment (SM1)
    sm1 = sum((abs(fft(image(:)))/length(image)).* (1:length(image)));

    % Peak Frequency (PKF)
    [~, pkf_idx] = max(abs(fft(image(:))));
    pkf = pkf_idx / length(image);

    % Instantaneous RMS Voltage
    instantaneous_rms = rms(image(:));

    % Average RMS Voltage
    average_rms = rms_val; % Assuming RMS value is an appropriate representation of average RMS

    % Combine features into a vector
    features = [mean_intensity, std_dev_intensity, range_intensity, ...
        median_intensity, interquartile_range, mean_bound, ...
        area, perimeter, compactness, mgr, vgr, rms_val, iemg, ssi, var_val, area_power, sm1, pkf, instantaneous_rms, average_rms];

    % Display the extracted features
    disp('************ FEATURES EXTRACTION 1 ***************');
    disp(['Mean = ' num2str(mean_intensity)]);
    disp(['Stdev = ' num2str(std_dev_intensity)]);
    disp(['Range = ' num2str(range_intensity)]);
    disp(['Median = ' num2str(median_intensity)]);
    disp(['Interquartile Range = ' num2str(interquartile_range)]);
    disp(['ModeROI = ' num2str(mode_intensity)]);
    % disp(['StdMode = ' num2str(std_mode)]);
    disp(['MeanBound = ' num2str(mean_bound)]);
    disp(['Area = ' num2str(area)]);
    disp(['Perimeter = ' num2str(perimeter)]);
    disp(['Compactness = ' num2str(compactness)]);
    % disp(['Skewness = ' num2str(skewness_val)]);
    % disp(['Kurtosis = ' num2str(kurtosis_val)]);
    disp(['MGR = ' num2str(mgr)]);
    disp(['VGR = ' num2str(vgr)]);
    disp(['RMS = ' num2str(rms_val)]);
    disp(['IEMG = ' num2str(iemg)]);
    disp(['SSI = ' num2str(ssi)]);
    disp(['VAR = ' num2str(var_val)]);
    disp(['Area of Power = ' num2str(area_power)]);
    %disp(['SM1 = ' num2str(sm1)]);
    disp(['PKF = ' num2str(pkf)]);
    disp(['Instantaneous RMS = ' num2str(instantaneous_rms)]);
    disp(['Average RMS = ' num2str(average_rms)]);
end
