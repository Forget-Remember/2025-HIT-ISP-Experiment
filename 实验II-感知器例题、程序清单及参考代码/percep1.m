% percep1.m (修正版本)
P = [-0.5 -0.5 0.3 0 1.0 0.1; -0.5 0.5 -0.5 1 0.8 -1]; % 2x5 输入矩阵 (2维特征 x 5个样本)
T = [1 1 0 0 0 1]; % 目标值：1x5 向量 (与样本数匹配)

% 初始化
[R, Q] = size(P);   % R=2 (输入维度), Q=5 (样本数)
S = size(T, 1);     % S=1 (输出维度)

% 权值初始化
W = rand(S, R) - 0.5;  % 替代 rands
b = rand(S, 1) - 0.5; 
B = repmat(b, 1, Q);   % 复制偏置向量 -> 1x5 矩阵

max_epoch = 20;
A = hardlim(W*P + B); % 初始网络输出

% 训练循环
converged = false;
for epoch = 1:max_epoch
    % 检查收敛
    if all(A == T)
        fprintf('收敛于 epoch %d\n', epoch);
        converged = true;
        break;
    end
    
    % 计算误差并直接更新权重 (核心数学公式)
    e = T - A;
    dW = e * P';
    db = e * ones(Q, 1);
    
    W = W + dW;
    b = b + db;
    B = repmat(b, 1, Q);  % 更新偏置矩阵
    A = hardlim(W*P + B); % 新输出
    
    % 每轮打印权重
    fprintf('Epoch %d: W=[%.4f, %.4f], b=%.4f\n', epoch, W(1), W(2), b);
end

if ~converged
    fprintf('未收敛（达到最大迭代次数 %d）\n', max_epoch);
end

% ==== 实验报告绘图部分 ====
figure;
hold on;
grid on;

% 1. 绘制样本点
pos = P(:, T==1); % 正类样本 (T=1)
neg = P(:, T==0); % 负类样本 (T=0)
scatter(pos(1,:), pos(2,:), 100, 'b', 'filled', 'DisplayName','正类 (T=1)');
scatter(neg(1,:), neg(2,:), 100, 'r', 'o', 'DisplayName','负类 (T=0)');

% 2. 绘制决策边界 (W1*x1 + W2*x2 + b = 0)
x_vals = [min(P(1,:))-0.5, max(P(1,:))+0.5];
y_vals = (-W(1)*x_vals - b) / W(2);  % 解方程：W1*x1 + W2*x2 + b = 0
plot(x_vals, y_vals, 'k--', 'LineWidth', 1.5, 'DisplayName','决策边界');

% 3. 标注信息
title(sprintf('感知机分类结果 (迭代次数: %d)', epoch));
xlabel('特征 x1'); ylabel('特征 x2');
legend('Location','best');

% 4. 计算并显示分类准确率
accuracy = sum(A == T) / Q * 100;
text(mean(x_vals), max(P(2,:))+0.5, ...
    sprintf('准确率: %.1f%%\nW=[%.4f, %.4f], b=%.4f', ...
    accuracy, W(1), W(2), b), ...
    'HorizontalAlignment','center');

% 5. 添加坐标轴
xlim([min(P(1,:))-0.7 max(P(1,:))+0.7]);
ylim([min(P(2,:))-0.7 max(P(2,:))+0.7]);
plot([0 0], ylim, 'k-', 'LineWidth', 0.5); % y轴
plot(xlim, [0 0], 'k-', 'LineWidth', 0.5); % x轴

hold off;