% bp6;例4.6
%
P=-1:0.1:1;
%P2=-1:0.1:1;
T=[-0.96 -0.577 -0.0729 0.377 0.641 0.66 0.461 0.1336 ...
   -0.201 -0.434 -0.5 -0.393 -0.1647 0.0988 0.3072 ...
   0.396 0.3449 0.1816 -0.0312 -0.2183 -0.3201];
net=newcf(minmax(P),[5,1],{'tansig','purelin'},'traingd');   %创建两层前向回馈网络

net.initParam = [];
%net.layers{1}.initFcn = 'initnw';           %第一层权值赋值函数
%net = init(net);                              %初始化网络
net = initnw(net,1);
w0 = net.iw{1,1},
b0 = net.b{1},
w1 = net.lw{2,1},
b1 = net.b{2},
disp('按任一键继续')
pause
y=sim(net,P);                                      %初始输出
plot(P,T,'ro')                                       %画输入矢量图 
hold on
plot(P,y)                                          %画输出矢量图
hold off

net.trainParam.epochs=7000;                         %初始化训练次数
net.trainParam.goal=9.5238e-004; %  sse=0.02;
net.iw{1,1}=w0*0.5;net.b{1}=b0*0.5;
net.trainParam.lr = 0.15;
[net tr]=train(net,P,T);                               %训练网络
Y=sim(net,P);                                      %计算结果

figure(2)
plot(P,T,'ro')                                       %画输入矢量图 
hold on
plot(P,Y)                                         %画输出矢量图
hold off

