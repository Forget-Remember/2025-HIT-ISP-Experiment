% 解决 rands 函数问题，改用 randn 初始化权重
P = -1:0.1:1;
T = [-0.96 -0.577 -0.0729 0.377 0.641 0.66 0.461 0.1336 ...
      -0.201 -0.434 -0.5 -0.393 -0.1647 0.0988 0.3072 ...
      0.396 0.3449 0.1816 -0.0312 -0.2183 -0.3201];

% 第一张图：原始数据点
figure(1);
plot(P, T, 'r+');
title('原始数据点');
xlabel('输入 P');
ylabel('目标 T');
saveas(gcf, '原始数据点.png');

% 初始化权重和偏置（替换 rands 函数）
[R, Q] = size(P);
S1 = 5;     % 隐藏层神经元数量
S2 = 1;     % 输出层神经元数量

% 使用 randn 替代 rands 进行初始化
W1 = randn(S1, R);  % 输入层到隐藏层权重
B1 = randn(S1, 1);  % 隐藏层偏置
W2 = randn(S2, S1); % 隐藏层到输出层权重
B2 = randn(S2, 1);  % 输出层偏置

% 前向传播（使用与训练时相同的输入范围）
P2 = P;
a1 = tansig(W1 * P2 + B1 * ones(1, Q));
A2 = purelin(W2 * a1 + B2 * ones(1, Q));

% 第二张图：初始随机权重下的预测
figure(2);
plot(P, T, 'r+', 'DisplayName', '目标值');
hold on;
plot(P, A2, 'b-', 'DisplayName', '初始预测');
title('初始随机权重下的预测');
xlabel('输入 P');
ylabel('输出');
legend('Location', 'Best');
saveas(gcf, '初始预测.png');

% 创建并训练神经网络
net = newff(minmax(P), [5, 1], {'tansig', 'purelin'}, 'traingd');
net.trainParam.epochs = 7000;
net.trainParam.goal = 9.5238e-4;  % SSE=0.02
net.trainParam.lr = 0.15;

[net, tr] = train(net, P, T);
Y = sim(net, P);

% 第三张图：训练后的预测
figure(3);
plot(P, T, 'r+', 'DisplayName', '目标值');
hold on;
plot(P, Y, 'g-', 'LineWidth', 1.5, 'DisplayName', '训练后预测');
title('神经网络训练结果');
xlabel('输入 P');
ylabel('输出');
legend('Location', 'Best');
grid on;
saveas(gcf, '训练结果.png');

% 第四张图：训练过程误差曲线
figure(4);
plot(tr.perf, 'LineWidth', 1.5);
title('训练误差曲线');
xlabel('训练次数');
ylabel('均方误差 (MSE)');
grid on;
saveas(gcf, '训练误差.png');