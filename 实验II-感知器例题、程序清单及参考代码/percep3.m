% percep3.m
%
P = [-0.5 -0.5 0.3 0 -0.8;-0.5 0.5 -0.5 1 0];
T = [1 1 0 0 0];
V = [-2 2 -2 2];
net = newp(minmax(P),1,'hardlim','learnp');    % 创建一个感知器网络
net.inputweights{1,1}.initFcn = 'rands';       % 赋输入权值的产生函数
net.biases{1}.initFcn = 'rands';               % 赋偏差的产生函数
net = init(net);                               % 初始化网络
W0 = net.iw{1,1};
B0 = net.b{1};
A = sim(net,P);                                % 计算网络输出
net.trainParam.epochs = 40;
[net, tr] = train(net,P,T);					 % 训练网络权值
W = net.iw{1,1};
B = net.b{1};
pause                               %	 看前面的网络训练结果图形，按任意键继续
plotpv(P,T,V);
hold on
plotpc(W0,B0)                                %做出分类线的曲线
plotpc(W,B)
hold off
fprintf ('\n Final Network Values : \n' )
W;
B;
fprintf('Tained for %.0f epochs',max(tr.epoch));
fprintf('\nNetwork classifies:');
if all ( hardlim ( W*P + B ) == T )
  disp ('Correctly.' )
else
  disp ('Incorrectly.' )
end
