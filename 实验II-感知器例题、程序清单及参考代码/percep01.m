%例2.1（解法2）
%percep01.m
net = newp([-1 1; -1 1],1);
P = [-0.5 -0.5 0.3 0;-0.5 0.5 -0.5 1];
T = [1 1 0 0];                         % 初始化
A = sim(net,P)					       % 训练前的网络输出
net.trainParam.epochs = 20;            % 定义最大循环次数
net = train(net,P,T);                  % 训练网络，使输出和期望相同
net.iw{1,1}                            % 输出训练后的网络权值
net.b{1}                               % 输出训练后的网络偏差
A = sim(net,P)                         % 训练后的网络输出