%% DATA GENERATION CONFIG

d = -6;
outer_rad = 16;
moon_width = 6;

total_points = 1400;
upper_moon_points = total_points / 2;
lower_moon_points = total_points / 2;

%% NEURAL NETWORK CONFIG

nn_layer = 2;
nn_layer_size = [1 1]; % last layer should always have one neuron
nn_layer_tf = {'tansig', 'purelin'};

wt_range = [-0.2 0.2];

%% TRAINING CONFIG

training_points = 1000;
validation_points = (total_points - training_points) / 2;
testing_points = (total_points - training_points) / 2;

training_function = 'traingda';
epochs = 1000;
learning_rate = 0.01;
fail_count = 10;
iterations = 20;

%% DATA GENERATION

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

%% NEURAL NETWORK BUILD

inputs = data(:, 1:2)';
targets = data(:, 3)';

net = network;

net.name = 'double moon classifier';
net.numInputs = 1;
net.numLayers = nn_layer;

% input
net.inputConnect = [1; zeros(nn_layer - 1, 1)];

%bias
net.biasConnect = ones(nn_layer, 1);

% hidden layers
for i = 1:nn_layer
    net.layers{i}.size = nn_layer_size(i);
    net.layers{i}.transferFcn = nn_layer_tf{i};
    if i == nn_layer
        break;
    else
        net.layerConnect(i + 1, i) = 1;
    end
end

% output layer
net.outputConnect = [zeros(1, nn_layer - 1) 1];

% configure
net = configure(net, inputs, targets);

%% NEURAL NETWORK CUSTOMIZE AND TRAIN

net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:training_points;
net.divideParam.valInd = training_points+1:training_points+validation_points;
net.divideParam.testInd = training_points+validation_points+1:total_points;

net.trainFcn = training_function;
net.trainParam.epochs = epochs;
net.trainParam.lr = learning_rate;
net.trainParam.max_fail = fail_count;

net.performFcn = 'mse';

net.plotFcns = {'plotconfusion', 'plotperform', 'plotregression'};

error_graph = zeros(iterations, 1);
epoch_graph = zeros(iterations, 1);

for k = 1:iterations
    net.IW{1, 1} = wt_range(1) + (wt_range(2) - wt_range(1)) * rand(nn_layer_size(1), 2);
    for i = 1:nn_layer
        net.b{i} = wt_range(1) + (wt_range(2) - wt_range(1)) * rand(nn_layer_size(i), 1);
        if i == nn_layer
            break;
        else
            net.LW{i + 1, i} = wt_range(1) + (wt_range(2) - wt_range(1)) * rand(nn_layer_size(i + 1), nn_layer_size(i));
        end
    end

    net = init(net);
    
    [net, tr] = train(net,inputs,targets);
    error_graph(k) = tr.best_perf;
    epoch_graph(k) = tr.num_epochs;
end

%% RESULTS

% resulting neural network design
view(net)

% resulting data points
figure(1)
plot(data(:, 1), data(:, 2), '.')
plotpc(net.IW{1,1},net.b{1})

% resultingg training and testing errors
figure(2)
plotperform(tr)

test_output = net(inputs(:, min(tr.testInd):max(tr.testInd)));

% resulting testing confusion matrix
figure(3)
plotconfusion(targets(min(tr.testInd):max(tr.testInd)), test_output)

% resulting average error and epochs
avg_testing_error = sum(error_graph) / length (error_graph)
avg_epochs = sum(epoch_graph) / length(epoch_graph)
