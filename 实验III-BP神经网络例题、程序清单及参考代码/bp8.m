% bp8;例4.8
%
P=[-6 -6.1 -4.1 -4 4 4.1 6 6.1];
T=[0.0 0.0 0.97 0.99 0.01 0.03 1 1];
net=newcf(minmax(P),[1],{'tansig'},'traingd');      
                                   %创建前向BP网络
net.iw{1,1};net.b{1};
net.trainParam.epochs=300;            %初始化训练次数
net.trainParam.lr=0.05;
[net tr]=train(net,P,T);                  %训练网络
w1=net.iw{1,1},b1=net.b{1},
Wrange=-1:0.1:1;Brange=-2:0.2:2;        %W值的行向量、B值的行向量
ES=errsurf(P,T,Wrange,Brange,'logsig');    %求单神经元的误差平面（只用在单神经元中）
mesh(ES,[60,30]);                     %作三维网状面，视角【60，30】 
title('Error Surface Graph')
xlabel('W');
ylabel('B');
zlabel('Error')
figure(2)
[C,h] =contour(Wrange,Brange,ES,6);     %作等高线图，ES为高
                                   %返回等高线矩阵C，列向量h是线或对象的句柄，
                                   %一条线一个句柄，这些被用作CLABEL的输入， 
                                   %每个对象包含每个等高线的高度 
clabel(C,h)                          %标上高度值
colormap cool                        %背景的颜色cool
hold on
plot(w1,b1,'r+')
xlabel('W');
ylabel('B');
hold off
figure(3)
plot(tr.perf)                           %画误差曲线
