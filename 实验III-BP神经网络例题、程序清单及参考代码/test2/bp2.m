%% 1. 数据准备与网络训练
P = [-3, 2];          % 输入向量
T = [0.4, 0.8];       % 目标输出

% 训练线性网络
net_linear = newlind(P, T);
w_linear = net_linear.iw{1,1};
b_linear = net_linear.b{1};
y_linear = sim(net_linear, P);

% 训练BP网络
net_bp = newcf(minmax(P), 1, {'tansig'});
net_bp.trainParam.epochs = 50;
net_bp.trainParam.goal = 0.001;
net_bp = train(net_bp, P, T);
w_bp = net_bp.iw{1,1};
b_bp = net_bp.b{1};
y_bp = sim(net_bp, P);

%% 2. 生成拟合曲线数据点
P_continuous = linspace(-4, 3, 100);  % 连续输入值

% 线性网络输出曲线
y_linear_curve = w_linear * P_continuous + b_linear;

% BP网络输出曲线
h_bp = tansig(w_bp * P_continuous + b_bp);  % 隐含层计算
y_bp_curve = purelin(h_bp);                 % 输出层计算

% ========= 图1：网络输出对比 =========
fig1 = figure('Position', [100, 100, 800, 600], 'Color', 'white');
hold on;

% 目标点
scatter(P, T, 100, 'r', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5)
text(P(1)+0.1, T(1), ' 目标点1 (-3,0.4)', 'FontSize', 10, 'FontWeight', 'bold')
text(P(2)+0.1, T(2), ' 目标点2 (2,0.8)', 'FontSize', 10, 'FontWeight', 'bold')

% 线性网络输出
plot(P_continuous, y_linear_curve, 'c-', 'LineWidth', 2)
plot(P, y_linear, 'c+', 'MarkerSize', 12, 'LineWidth', 2)
text(P(1), y_linear(1)-0.15, sprintf('线性输出: %.4f', y_linear(1)), 'Color', 'c', 'FontSize', 10, 'BackgroundColor', 'white')
text(P(2), y_linear(2)+0.15, sprintf('线性输出: %.4f', y_linear(2)), 'Color', 'c', 'FontSize', 10, 'BackgroundColor', 'white')

% BP网络输出
plot(P_continuous, y_bp_curve, 'b-', 'LineWidth', 1.5)
plot(P, y_bp, 'bo', 'MarkerSize', 10, 'LineWidth', 2)
text(P(1), y_bp(1)+0.15, sprintf('BP输出: %.4f', y_bp(1)), 'Color', 'b', 'FontSize', 10, 'BackgroundColor', 'white')
text(P(2), y_bp(2)-0.15, sprintf('BP输出: %.4f', y_bp(2)), 'Color', 'b', 'FontSize', 10, 'BackgroundColor', 'white')

% 图例和标注
legend({'目标点', '线性拟合曲线', '线性网络输出点', 'BP网络拟合曲线', 'BP网络输出点'}, ...
       'Location', 'southeast', 'FontSize', 10)
title('线性网络与BP网络性能对比 (例4.2)', 'FontSize', 12, 'FontWeight', 'bold')
xlabel('输入 P', 'FontSize', 11)
ylabel('输出 Y', 'FontSize', 11)
grid on
axis([-4 3 -1 1.5])
set(gca, 'FontSize', 10, 'LineWidth', 1.5)
box on

% 保存图片
exportgraphics(fig1, fullfile(save_dir, '实验结果', '网络性能对比.png'), 'Resolution', 300);
close(fig1);

% ========= 图2：误差分析 =========
fig2 = figure('Position', [100, 100, 900, 400], 'Color', 'white');

subplot(1,2,1)
% 线性网络误差
bar(1:2, abs(T - y_linear), 'FaceColor', [0.2, 0.8, 0.8], 'EdgeColor', 'k')
hold on
plot([0,3], [0,0], 'k--')
title('线性网络绝对误差', 'FontSize', 11, 'FontWeight', 'bold')
ylabel('|误差|', 'FontSize', 10)
set(gca, 'XTick', [1,2], 'XTickLabel', {'点1(-3)','点2(2)'}, 'FontSize', 10)
ylim([0 0.035])
text(1, abs(T(1)-y_linear(1))+0.003, sprintf('%.4f', abs(T(1)-y_linear(1))), ...
    'HorizontalAlignment', 'center', 'FontSize', 10)
grid on
box on

subplot(1,2,2)
% BP网络误差
bar(1:2, abs(T - y_bp), 'FaceColor', [0.2, 0.4, 0.8], 'EdgeColor', 'k')
hold on
plot([0,3], [0,0], 'k--')
title('BP网络绝对误差', 'FontSize', 11, 'FontWeight', 'bold')
set(gca, 'XTick', [1,2], 'XTickLabel', {'点1(-3)','点2(2)'}, 'FontSize', 10)
ylim([0 0.035])
text(1, abs(T(1)-y_bp(1))+0.003, sprintf('%.4f', abs(T(1)-y_bp(1))), ...
    'HorizontalAlignment', 'center', 'FontSize', 10)
text(2, abs(T(2)-y_bp(2))+0.003, sprintf('%.4f', abs(T(2)-y_bp(2))), ...
    'HorizontalAlignment', 'center', 'FontSize', 10)
grid on
box on

% 添加整体标题
sgtitle('网络输出误差分析', 'FontSize', 12, 'FontWeight', 'bold');

% 保存图片
exportgraphics(fig2, fullfile(save_dir, '实验结果', '网络误差分析.png'), 'Resolution', 300);
close(fig2);

%% 3. 显示结果
fprintf('实验图表已保存至文件夹: %s\n', fullfile(save_dir, '实验结果'));
fprintf('1. 网络性能对比图: 网络性能对比.png\n');
fprintf('2. 误差分析图: 网络误差分析.png\n');

% 在窗口中显示关键参数
disp('===== 网络参数对比 =====')
disp('网络类型      权重(w)     偏置(b)    ')
fprintf('线性网络     %-9.4f    %-9.4f\n', w_linear, b_linear);
fprintf('BP网络       %-9.4f    %-9.4f\n', w_bp, b_bp);

disp('===== 输出误差对比 =====')
disp('样本点     目标值     线性输出    BP输出     线性误差    BP误差')
fprintf('点1(-3)   %-9.4f  %-9.4f  %-9.4f  %-9.4f  %-9.4f\n', ...
        T(1), y_linear(1), y_bp(1), abs(T(1)-y_linear(1)), abs(T(1)-y_bp(1)));
fprintf('点2(2)    %-9.4f  %-9.4f  %-9.4f  %-9.4f  %-9.4f\n', ...
        T(2), y_linear(2), y_bp(2), abs(T(2)-y_linear(2)), abs(T(2)-y_bp(2)));