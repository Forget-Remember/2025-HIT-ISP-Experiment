% ===== MnistFC 函数：3层全连接的前向+反向+更新 =====
function [W1, W5, Wo] = MnistFC(W1, W5, Wo, X, D)
  alpha = 0.01;
  beta  = 0.95;

  momentum1 = zeros(size(W1));
  momentum5 = zeros(size(W5));
  momentumo = zeros(size(Wo));

  N = length(D);
  bsize = 100;
  blist = 1:bsize:(N - bsize + 1);

  for batch = 1:length(blist)
    dW1 = zeros(size(W1));
    dW5 = zeros(size(W5));
    dWo = zeros(size(Wo));

    % 小批次循环
    begin = blist(batch);
    for k = begin:(begin + bsize - 1)
      % 前向传播
      x  = X(:, :, k);               % 28×28
      x_flat = reshape(x, [], 1);    % 784×1

      v1 = W1 * x_flat;              % 2000×1
      y1 = ReLU(v1);

      v2 = W5 * y1;                  % 100×1
      y2 = ReLU(v2);

      v  = Wo * y2;                  % 10×1
      y  = Softmax(v);

      % 构造one-hot向量
      d = zeros(10, 1);
      d(D(k)) = 1;

      % 反向传播
      e      = d - y;                % 输出误差 10×1
      delta3 = e;                    % 最后一层直接等于误差

      e2     = Wo' * delta3;         % 反传到隐藏2 (100×1)
      delta2 = (y2 > 0) .* e2;       % ReLU导数

      e1     = W5' * delta2;         % 反传到隐藏1 (2000×1)
      delta1 = (y1 > 0) .* e1;       % ReLU导数

      % 梯度累加
      dW1 = dW1 + delta1 * x_flat';
      dW5 = dW5 + delta2 * y1';
      dWo = dWo + delta3 * y2';
    end

    % 取平均
    dW1 = dW1 / bsize;
    dW5 = dW5 / bsize;
    dWo = dWo / bsize;

    % 动量更新
    momentum1 = alpha * dW1 + beta * momentum1;
    W1        = W1 + momentum1;

    momentum5 = alpha * dW5 + beta * momentum5;
    W5        = W5 + momentum5;

    momentumo = alpha * dWo + beta * momentumo;
    Wo        = Wo + momentumo;
  end
end