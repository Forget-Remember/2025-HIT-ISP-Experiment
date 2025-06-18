% 3层全连接网络实现手写数字识别
clear all

addpath(genpath('..\'))
% 加载MNIST测试集（这里用作示例，训练时只取前8000张）
Images = loadMNISTImages('../MNIST/t10k-images.idx3-ubyte');
Images = reshape(Images, 28, 28, []);   % 28×28×10000
Labels = loadMNISTLabels('../MNIST/t10k-labels.idx1-ubyte');
Labels(Labels == 0) = 10;    % 把数字0映射到标签10

% figure,imagesc(Images(:,:,6510))

rng(1);

% ===== 初始化3个全连接层的权重 =====
% W1:  输入 28×28=784 → 隐藏单元 2000
W1 = 1e-2 * randn([2000, 784]);

% W5:  隐藏1 (2000) → 隐藏2 (100)
W5 = (2*rand(100, 2000) - 1) * sqrt(6) / sqrt(360 + 2000);

% Wo:  隐藏2 (100) → 输出 (10)
Wo = (2*rand(10, 100) - 1) * sqrt(6) / sqrt(10 + 100);

% 训练集：前8000张
X = Images(:, :, 1:8000);
D = Labels(1:8000);

% 训练3个epoch
for epoch = 1:3
  fprintf('Epoch %d\n', epoch);
  [W1, W5, Wo] = MnistFC(W1, W5, Wo, X, D);
end

save('MnistFC.mat');

% ===== 测试 =====
X = Images(:, :, 8001:10000);
D = Labels(8001:10000);

acc = 0;
N   = length(D);
for k = 1:N
  x = X(:, :, k);                  % 输入 28×28
  x_flat = reshape(x, [], 1);      % 784×1

  % 前向传播：3层全连接
  v1 = W1 * x_flat;                % 2000×1
  y1 = ReLU(v1);

  v2 = W5 * y1;                    % 100×1
  y2 = ReLU(v2);

  v  = Wo * y2;                    % 10×1
  y  = Softmax(v);

  [~, i] = max(y);
  if i == D(k)
    acc = acc + 1;
  end
end

acc = acc / N;
fprintf('Accuracy is %f\n', acc);