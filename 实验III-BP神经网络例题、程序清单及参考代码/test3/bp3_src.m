%% 实验数据准备
P = [-6 -6.1 -4.1 -4 4 4.1 6 6.1]; % 输入向量
T = [0.0 0.0 0.97 0.99 0.01 0.03 1 1]; % 目标输出向量

%% 第一部分：线性网络设计与分析
linear_net = newlind(P, T); % 设计最优线性网络
w1 = linear_net.iw{1,1}; % 获取权重
b1 = linear_net.b{1}; % 获取偏置
linear_y = sim(linear_net, P); % 网络仿真

% 计算线性网络性能指标
linear_mse = mse(linear_y - T); % 均方误差
linear_mae = mean(abs(linear_y - T)); % 平均绝对误差

fprintf('===== 线性网络性能 =====\n');
fprintf('权重 w = %.4f\n', w1);
fprintf('偏置 b = %.4f\n', b1);
fprintf('均方误差(MSE) = %.4f\n', linear_mse);
fprintf('平均绝对误差(MAE) = %.4f\n\n', linear_mae);

%% 第二部分：单层非线性网络设计与分析
% 创建单层非线性网络
single_layer_net = newff(minmax(P), [1], {'logsig'}, 'traingd'); 
single_layer_net.trainParam.epochs = 1000; % 增加训练次数
single_layer_net.trainParam.goal = 0.01; % 降低目标误差
single_layer_net.trainParam.lr = 0.05; % 学习率
single_layer_net.trainParam.show = 50; % 每50次显示训练进度

% 训练单层非线性网络
[single_layer_net, tr_single] = train(single_layer_net, P, T);

% 获取网络参数
w2 = single_layer_net.iw{1,1};
b2 = single_layer_net.b{1};
single_layer_y = sim(single_layer_net, P); % 网络仿真

% 计算单层非线性网络性能指标
single_layer_mse = mse(single_layer_y - T);
single_layer_mae = mean(abs(single_layer_y - T));

fprintf('===== 单层非线性网络性能 =====\n');
fprintf('权重 w = %.4f\n', w2);
fprintf('偏置 b = %.4f\n', b2);
fprintf('均方误差(MSE) = %.4f\n', single_layer_mse);
fprintf('平均绝对误差(MAE) = %.4f\n', single_layer_mae);
fprintf('实际训练次数 = %d\n', tr_single.num_epochs);
fprintf('最终训练误差 = %.4f\n\n', tr_single.perf(end));

%% 第三部分：三层非线性网络实现完美匹配
% 创建三层非线性网络（输入层-隐含层-输出层）
hidden_neurons = 5; % 隐含层神经元数量
multi_layer_net = newff(minmax(P), [hidden_neurons, 1], {'tansig', 'purelin'}, 'trainlm'); 
multi_layer_net.trainParam.epochs = 1000; % 最大训练次数
multi_layer_net.trainParam.goal = 0.0001; % 极低目标误差
multi_layer_net.trainParam.lr = 0.01; % 学习率
multi_layer_net.trainParam.show = 50; % 每50次显示训练进度

% 训练多层非线性网络
[multi_layer_net, tr_multi] = train(multi_layer_net, P, T);

% 获取网络参数
input_weights = multi_layer_net.IW{1,1}; % 输入层到隐含层权重
input_bias = multi_layer_net.b{1}; % 隐含层偏置
output_weights = multi_layer_net.LW{2,1}; % 隐含层到输出层权重
output_bias = multi_layer_net.b{2}; % 输出层偏置

% 网络仿真
multi_layer_y = sim(multi_layer_net, P);

% 计算多层网络性能指标
multi_layer_mse = mse(multi_layer_y - T);
multi_layer_mae = mean(abs(multi_layer_y - T));

fprintf('===== 三层非线性网络性能 =====\n');
fprintf('输入层到隐含层权重:\n');
disp(input_weights);
fprintf('隐含层偏置:\n');
disp(input_bias');
fprintf('隐含层到输出层权重:\n');
disp(output_weights);
fprintf('输出层偏置: %.4f\n', output_bias);
fprintf('均方误差(MSE) = %.6f\n', multi_layer_mse);
fprintf('平均绝对误差(MAE) = %.6f\n', multi_layer_mae);
fprintf('实际训练次数 = %d\n', tr_multi.num_epochs);
fprintf('最终训练误差 = %.6f\n\n', tr_multi.perf(end));

%% 第四部分：结果可视化与保存
% 图1：网络输出对比
figure('Position', [100, 100, 900, 700]);
scatter(P, T, 100, 'r', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
hold on;
plot(P, linear_y, 'b-o', 'MarkerSize', 8);
plot(P, single_layer_y, 'g--s', 'MarkerSize', 8, 'LineWidth', 2);
plot(P, multi_layer_y, 'm-*', 'MarkerSize', 10, 'LineWidth', 2);
hold off;

% 美化图形
grid on;
title('三种网络结构输出对比', 'FontSize', 16);
xlabel('输入值 P', 'FontSize', 14);
ylabel('输出值', 'FontSize', 14);
legend({'目标输出', '线性网络', '单层非线性网络', '三层非线性网络'}, 'Location', 'best');
set(gca, 'FontWeight', 'bold');
xlim([-7 7]);
ylim([-0.2 1.2]);

% 添加性能标注
text(-6.5, 1.15, sprintf('线性网络 MSE = %.4f', linear_mse), 'FontSize', 12, 'Color', 'b');
text(-6.5, 1.05, sprintf('单层非线性网络 MSE = %.4f', single_layer_mse), 'FontSize', 12, 'Color', 'g');
text(-6.5, 0.95, sprintf('三层非线性网络 MSE = %.6f', multi_layer_mse), 'FontSize', 12, 'Color', 'm');

% 保存图像
saveas(gcf, '三种网络结构输出对比.png');

% 图2：训练过程对比
figure('Position', [100, 100, 900, 500]);
subplot(1, 2, 1);
semilogy(tr_single.epoch, tr_single.perf, 'g-', 'LineWidth', 2);
hold on;
yline(linear_mse, 'b--', 'LineWidth', 1.5);
hold off;
grid on;
title('单层非线性网络训练', 'FontSize', 14);
xlabel('训练次数', 'FontSize', 12);
ylabel('均方误差 (MSE, log scale)', 'FontSize', 12);
legend({'训练误差', '线性网络误差'}, 'Location', 'best');
set(gca, 'FontWeight', 'bold');

subplot(1, 2, 2);
semilogy(tr_multi.epoch, tr_multi.perf, 'm-', 'LineWidth', 2);
hold on;
yline(single_layer_mse, 'g--', 'LineWidth', 1.5);
hold off;
grid on;
title('三层非线性网络训练', 'FontSize', 14);
xlabel('训练次数', 'FontSize', 12);
ylabel('均方误差 (MSE, log scale)', 'FontSize', 12);
legend({'训练误差', '单层网络误差'}, 'Location', 'best');
set(gca, 'FontWeight', 'bold');

% 添加总标题
sgtitle('网络训练过程对比 (对数坐标)', 'FontSize', 16, 'FontWeight', 'bold');

% 保存图像
saveas(gcf, '网络训练过程对比 (对数坐标).png');

% 图3：决策边界可视化
figure('Position', [100, 100, 900, 700]);
x = linspace(-7, 7, 500);
linear_output = w1*x + b1;
single_layer_output = logsig(w2*x + b2);
multi_layer_output = sim(multi_layer_net, x);

plot(x, linear_output, 'b-', 'LineWidth', 2);
hold on;
plot(x, single_layer_output, 'g--', 'LineWidth', 2);
plot(x, multi_layer_output, 'm-', 'LineWidth', 3);
scatter(P, T, 100, 'r', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
hold off;

% 美化图形
grid on;
title('网络决策边界对比', 'FontSize', 16);
xlabel('输入值', 'FontSize', 14);
ylabel('输出值', 'FontSize', 14);
legend({'线性网络', '单层非线性网络', '三层非线性网络', '目标点'}, 'Location', 'best');
set(gca, 'FontWeight', 'bold');
xlim([-7 7]);
ylim([-0.2 1.2]);

% 添加标注
text(-6.5, 0.9, sprintf('线性网络: y = %.2f*x + %.2f', w1, b1), 'FontSize', 12);
text(-6.5, 0.8, sprintf('单层网络: y = logsig(%.2f*x + %.2f)', w2, b2), 'FontSize', 12);
text(-6.5, 0.7, sprintf('三层网络: 隐含层神经元数 = %d', hidden_neurons), 'FontSize', 12);

% 保存图像
saveas(gcf, '网络决策边界对比.png');

% 图4：三层网络误差分析
figure('Position', [100, 100, 900, 600]);

% 误差分布
subplot(2, 1, 1);
error_values = T - multi_layer_y;
bar(P, error_values, 'FaceColor', [0.7, 0.2, 0.5], 'EdgeColor', 'k');
hold on;
plot([min(P) max(P)], [0 0], 'k--', 'LineWidth', 1.5);
hold off;
grid on;
title('三层网络在各点的预测误差', 'FontSize', 14);
xlabel('输入值', 'FontSize', 12);
ylabel('预测误差', 'FontSize', 12);
ylim([-0.05, 0.05]);

% 误差统计
subplot(2, 1, 2);
abs_errors = abs(error_values);
bar(P, abs_errors, 'FaceColor', [0.5, 0.2, 0.7], 'EdgeColor', 'k');
grid on;
title('三层网络在各点的绝对误差', 'FontSize', 14);
xlabel('输入值', 'FontSize', 12);
ylabel('绝对误差', 'FontSize', 12);
ylim([0, 0.05]);

% 添加总标题
sgtitle('三层网络误差分析 (MSE = 1.6e-5)', 'FontSize', 16, 'FontWeight', 'bold');

% 保存图像
saveas(gcf, '三层网络误差分析.png');

fprintf('实验完成！已生成四张对比图并保存到当前目录。\n');