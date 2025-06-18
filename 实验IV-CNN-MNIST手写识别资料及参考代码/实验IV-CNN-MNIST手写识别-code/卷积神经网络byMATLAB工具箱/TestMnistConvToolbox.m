%% 使用 MATLAB 工具箱构建 CNN 手写数字识别
clear; close all; clc;
rng(1)

addpath(genpath('..\'))
%% 1) 加载并准备 MNIST 数据
Images = loadMNISTImages('../MNIST/t10k-images.idx3-ubyte');
% loadMNISTImages 会返回 28×28×N 的矩阵，符合 imageInputLayer 要求
Images = reshape(Images, 28, 28, 1, []);    % 28×28×1×10000
Labels = loadMNISTLabels('../MNIST/t10k-labels.idx1-ubyte');
Labels(Labels == 0) = 10;
LabelsCat = categorical(Labels);

% 划分训练/测试集：前 8000 张训练，后 2000 张测试
XTrain = Images(:, :, 1, 1:8000);
YTrain = LabelsCat(1:8000);
XTest  = Images(:, :, 1, 8001:10000);
YTest  = LabelsCat(8001:10000);

%% 2) 定义网络结构（与自实现版本对应）
layers = [
    imageInputLayer([28 28 1], "Name", "input")
    
    convolution2dLayer([9,9], 20, "Padding", 0, "Name", "conv1")  % 'Padding',0 等价于 valid
    reluLayer("Name", "relu1")
    maxPooling2dLayer(2, "Stride", 2, "Name", "pool1")
    
    fullyConnectedLayer(100, "Name", "fc1")                   % 10×10×20 = 2000 → 100
    reluLayer("Name", "relu2")
    
    fullyConnectedLayer(10, "Name", "fc_out")
    softmaxLayer("Name", "softmax")
    classificationLayer("Name", "classOutput")
    ];

%% 3) 配置训练选项
options = trainingOptions("adam", ...
    InitialLearnRate    = 0.01, ...             % 初始学习率
    LearnRateSchedule   = "piecewise", ...      % 使用分段式学习率衰减
    LearnRateDropFactor = 0.5, ...              % 衰减倍数
    LearnRateDropPeriod = 1, ...                % 衰减周期（epoch）
    MaxEpochs           = 3, ...
    MiniBatchSize       = 100, ...
    Shuffle             = "every-epoch", ...
    ValidationFrequency = 1, ...
    Verbose             = false, ...
    Plots               = "training-progress");

%% 4) 训练网络
netCNN = trainNetwork(XTrain, YTrain, layers, options);

%% 5) 测试与评估
YPred = classify(netCNN, XTest);
accuracy = sum(YPred == YTest)/numel(YTest);
fprintf("测试集准确率: %.4f\n", accuracy);
