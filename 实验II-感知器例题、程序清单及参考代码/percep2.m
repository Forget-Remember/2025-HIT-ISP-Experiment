% percep2.m
%
% 初始化、赋值
P = [0.1 0.7 0.8 0.8 1.0 0.3 0.0 -0.3 -0.5 -1.5;
    1.2 1.8 1.6 0.6 0.8 0.5 0.2 0.8 -1.5 -1.3];
T = [1 1 1 0 0 1 1 1 0 0;
0 0 0 0 0 1 1 1 1 1];  
[ R, ~ ] = size (P);	
[ S, Q ] = size (T);	
net = newp(minmax(P),S);		% 建立一个感知器网络
[ W0, B0 ] = rands (S, R);
net.iw{1,1} = W0;
net.b{1} = B0;
net.trainParam.epochs = 20;      % 定义最大循环次数
net = train(net,P,T);
% 绘制训练后的分类结果
V = [ -2 2 -2 2 ];				% 取一数组限制坐标数值大小
plotpv ( P, T, V );				% 在输入矢量空间绘画输入矢量和目标矢量的位置
axis ('equal'),					% 令横坐标和纵坐标等距离长度
title ('Input Vector Graph'),		% 写图标题
xlabel ('p1'),			   	    % 写横轴标题
ylabel ('p2'),				    % 写纵轴标题
plotpc (net.iw{1,1}, net.b{1} );	% 绘制由W和B在输入平面中形成的最终分类线
