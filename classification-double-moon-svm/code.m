%% CONFIG

c_value = {0.01, 1, 100};
kernel = {'linear', 'poly', 'rbf'};

d = -12;
outer_rad = 16;
moon_width = 6;

total_points = 1400;
training_points = 1000;

%% DATA GENERATION

upper_moon_points = total_points / 2;
lower_moon_points = total_points / 2;

upper_moon_index = 0;
lower_moon_index = 0;
total_index = 0;

data = zeros(total_points, 3);

while true
    
    % standard origin
    y = - (outer_rad + d) + (2 * outer_rad + d) * rand(1);
    x = - (outer_rad) + (3 * outer_rad - moon_width / 2) * rand(1);
    
    [theta1, R1] = cart2pol(x, y);
    Dtheta1 = rad2deg(theta1);
    
    % shifted origin
    shifted_y = y + d;
    shifted_x = x - (outer_rad - moon_width / 2);
    
    [theta2, R2] = cart2pol(shifted_x, shifted_y);
    Dtheta2 = rad2deg(theta2);
    
    % lies in upper moon
    if upper_moon_index < upper_moon_points
        if ((Dtheta1 >= 0) && (Dtheta1 <= 180)) && ((R1 <= outer_rad) && (R1 >= (outer_rad - moon_width)))
            upper_moon_index = upper_moon_index + 1;
            total_index = total_index + 1;
            data(total_index, :) = [x y 1];
        end
    % lies in lower moon
    elseif lower_moon_index < lower_moon_points
        if ((Dtheta2 <= 0) && (Dtheta2 >= -180)) && ((R2 <= outer_rad) && (R2 >= (outer_rad - moon_width)))
            lower_moon_index = lower_moon_index + 1;
            total_index = total_index + 1;
            data(total_index, :) = [x y 0];
        end
    end

    % enough data points found
    if upper_moon_index == upper_moon_points && lower_moon_index == lower_moon_points
        break;
    end
end

data = data(randperm(size(data, 1)), :);    %randomize data order

%% TRAINING AND TESTING DATA

training_inputs = data(1:training_points, 1:2);
training_targets = data(1:training_points, 3);

testing_inputs = data(training_points+1:total_points, 1:2);
testing_targets = data(training_points+1:total_points, 3);

l = 0.2;
[x1Grid,x2Grid] = meshgrid(min(data(:,1)):l:max(data(:,1)),min(data(:,2)):l:max(data(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];

%% MODEL GENERATION AND BOUNDARY PLOTTING

SVMModel = cell(length(c_value),length(kernel));

t = 1;
for i = 1 : length(c_value)
    for j = 1 : length(kernel)
        SVMModel{i, j} = fitcsvm(training_inputs, training_targets, 'KernelFunction', kernel{j}, 'Standardize', true, 'BoxConstraint', c_value{i});
        
        [~,scores] = predict(SVMModel{i, j}, xGrid);
        
        subplot(length(c_value), length(kernel), t);
        gscatter(data(:,1),data(:,2),data(:,3),'rb','.', [],'off');
        hold on
        contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'k');
        axis equal
        hold off
        title("c=" + c_value{i} + ", kernel: " + kernel{j})
        t = t + 1;
    end
end

%% CONFUSION  MATRIX
% NOTE: Since MATLAB doesn't allow subplotting confusion matrices, the following code will
% generate all confusion matrices separately

%{
t = 1;
for i = 1 : length(c_value)
    for j = 1 : length(kernel)
        figure(t)
        testing_obtained = predict(SVMModel{i, j}, testing_inputs);
        plotconfusion(testing_targets', testing_obtained', "c=" + c_value{i} + ", kernel: " + kernel{j});
        t = t + 1;
    end
end
%}