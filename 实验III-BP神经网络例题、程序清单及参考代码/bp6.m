% bp6;��4.6
%
P=-1:0.1:1;
%P2=-1:0.1:1;
T=[-0.96 -0.577 -0.0729 0.377 0.641 0.66 0.461 0.1336 ...
   -0.201 -0.434 -0.5 -0.393 -0.1647 0.0988 0.3072 ...
   0.396 0.3449 0.1816 -0.0312 -0.2183 -0.3201];
net=newcf(minmax(P),[5,1],{'tansig','purelin'},'traingd');   %��������ǰ���������

net.initParam = [];
%net.layers{1}.initFcn = 'initnw';           %��һ��Ȩֵ��ֵ����
%net = init(net);                              %��ʼ������
net = initnw(net,1);
w0 = net.iw{1,1},
b0 = net.b{1},
w1 = net.lw{2,1},
b1 = net.b{2},
disp('����һ������')
pause
y=sim(net,P);                                      %��ʼ���
plot(P,T,'ro')                                       %������ʸ��ͼ 
hold on
plot(P,y)                                          %�����ʸ��ͼ
hold off

net.trainParam.epochs=7000;                         %��ʼ��ѵ������
net.trainParam.goal=9.5238e-004; %  sse=0.02;
net.iw{1,1}=w0*0.5;net.b{1}=b0*0.5;
net.trainParam.lr = 0.15;
[net tr]=train(net,P,T);                               %ѵ������
Y=sim(net,P);                                      %������

figure(2)
plot(P,T,'ro')                                       %������ʸ��ͼ 
hold on
plot(P,Y)                                         %�����ʸ��ͼ
hold off

