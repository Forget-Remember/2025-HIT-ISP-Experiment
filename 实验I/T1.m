%% 实验初始化
clear; close all; clc;
rng(42); % 固定随机种子

%% 数据生成（含噪声的二次函数）
n_train = 10;   % 训练数据点数
n_test = 100;   % 测试数据点数
noise_std = 0.5; % 噪声标准差

% 生成训练数据
x_train = linspace(0, 1, n_train)';
y_true = 2*x_train.^2 - 3*x_train + 1;
y_train = y_true + noise_std*randn(n_train,1);

% 生成测试数据
x_test = linspace(0, 1, n_test)';
y_test = 2*x_test.^2 - 3*x_test + 1;

%% 模型训练与评估（不同多项式阶数）
degrees = [1, 2, 9];
colors = ['r', 'g', 'b'];
figure('Position', [100,100,1200,400]);

for i = 1:3
    p = degrees(i);
    
    % 构建设计矩阵
    X_train = zeros(n_train, p+1);
    for j = 0:p
        X_train(:,j+1) = x_train.^j;
    end
    
    % 正规方程求解权重
    w = (X_train'*X_train)\(X_train'*y_train);
    
    % 训练集预测
    y_pred_train = X_train*w;
    mse_train = mean((y_pred_train - y_train).^2);
    
    % 测试集预测
    X_test = zeros(n_test, p+1);
    for j = 0:p
        X_test(:,j+1) = x_test.^j;
    end
    y_pred_test = X_test*w;
    mse_test = mean((y_pred_test - y_test).^2);
    
    % 可视化结果
    subplot(1,3,i);
    xx = linspace(0,1,100)';
    XX = zeros(100,p+1);
    for j = 0:p
        XX(:,j+1) = xx.^j;
    end
    plot(xx, XX*w, 'Color', colors(i), 'LineWidth', 2);
    hold on;
    scatter(x_train, y_train, 100, 'k', 'filled');
    title(sprintf('%d阶多项式 (MSE: %.4f)', p, mse_test));
    xlabel('x'); ylabel('y');
    grid on;
    axis([0 1 -3 3]);
    
    % 显示权重参数
    text(0.05, 2.5, sprintf('权重参数:\nw = [%s]', ...
        sprintf('%.2f ', w)), 'FontSize', 9);
end

% 保存第一个figure
saveas(gcf, 'poly_fit_results.png');
print('poly_fit_results', '-dpng', '-r300');

%% K折交叉验证实现（选择最佳多项式阶数）
k = 5;
max_degree = 10;
cv_mse = zeros(max_degree,1);

% 数据打乱
idx = randperm(n_train);
x_cv = x_train(idx);
y_cv = y_train(idx);

for p = 1:max_degree
    mse_sum = 0;
    
    % K折交叉验证
    for fold = 1:k
        % 划分训练/验证集
        val_idx = (fold-1)*floor(n_train/k)+1 : fold*floor(n_train/k);
        train_idx = setdiff(1:n_train, val_idx);
        
        % 构建矩阵
        X_tr = zeros(length(train_idx), p+1);
        X_val = zeros(length(val_idx), p+1);
        for j = 0:p
            X_tr(:,j+1) = x_cv(train_idx).^j;
            X_val(:,j+1) = x_cv(val_idx).^j;
        end
        
        % 训练模型
        w = (X_tr'*X_tr)\(X_tr'*y_cv(train_idx));
        
        % 验证集预测
        y_pred = X_val*w;
        mse_sum = mse_sum + mean((y_pred - y_cv(val_idx)).^2);
    end
    
    cv_mse(p) = mse_sum/k;
end

% 可视化交叉验证结果
figure;
plot(1:max_degree, cv_mse, 'bo-', 'LineWidth', 1.5);
xlabel('多项式阶数');
ylabel('交叉验证MSE');
title('K折交叉验证结果');
grid on;
hold on;
[~, best_degree] = min(cv_mse);
plot(best_degree, cv_mse(best_degree), 'r*', 'MarkerSize', 15);
legend('验证误差', '最佳阶数');

% 保存第二个figure
saveas(gcf, 'cross_validation.png');
print('cross_validation', '-dpng', '-r300');

%% 理论概念分析
% 模型容量分析
figure;
degrees = 1:10;
train_mse = zeros(size(degrees));
test_mse = zeros(size(degrees));

for p = degrees
    X_train = zeros(n_train, p+1);
    for j = 0:p
        X_train(:,j+1) = x_train.^j;
    end
    w = (X_train'*X_train)\(X_train'*y_train);
    y_pred = X_train*w;
    train_mse(p) = mean((y_pred - y_train).^2);
    
    X_test = zeros(n_test, p+1);
    for j = 0:p
        X_test(:,j+1) = x_test.^j;
    end
    y_pred = X_test*w;
    test_mse(p) = mean((y_pred - y_test).^2);
end

plot(degrees, train_mse, 'b-o', degrees, test_mse, 'r-s');
xlabel('模型复杂度（多项式阶数）');
ylabel('MSE');
legend('训练误差', '测试误差');
title('偏差-方差权衡');
grid on;

% 保存第三个figure
saveas(gcf, 'bias_variance_tradeoff.png');
print('bias_variance_tradeoff', '-dpng', '-r300');