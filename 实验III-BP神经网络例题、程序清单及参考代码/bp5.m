%bp5;例4.5
%
time=[];err=[];epo=[];

for i=1:6                                   %分别进行取S1= 1 - 6节点时的网络训练
t(i)=cputime;                                      %起始CPU时间
P=[0 0 1 1;0 1 0 1];
T=[0 1 1 0];
net=newcf(minmax(P),[i,1],{'tansig' 'purelin'},'traingd' ); 
                                                 %创建两层前向BP网络
net.trainParam.show = 50;                            %每50次显示一次结果
net.trainParam.lr = 0.05;                           %学习参数
net.trainParam.epochs = 300;                       %最大循环参数
[net,tr]=train(net,P,T);                            %训练网络
Y=sim(net,P);                                   %计算输出结果
err=[err;tr.perf];
time(i)=cputime-t(i);                             %计算程序运行时间
end

plot(err(1,:));                                   %做出以上6个网络的训练误差图    
hold on
plot(err(2,:),'g:');
plot(err(3,:),'g-');
plot(err(4,:),':');
plot(err(5,:),'r:');
plot(err(6,:),'r-');
hold off
time
