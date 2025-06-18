% bp8;��4.8
%
P=[-6 -6.1 -4.1 -4 4 4.1 6 6.1];
T=[0.0 0.0 0.97 0.99 0.01 0.03 1 1];
net=newcf(minmax(P),[1],{'tansig'},'traingd');      
                                   %����ǰ��BP����
net.iw{1,1};net.b{1};
net.trainParam.epochs=300;            %��ʼ��ѵ������
net.trainParam.lr=0.05;
[net tr]=train(net,P,T);                  %ѵ������
w1=net.iw{1,1},b1=net.b{1},
Wrange=-1:0.1:1;Brange=-2:0.2:2;        %Wֵ����������Bֵ��������
ES=errsurf(P,T,Wrange,Brange,'logsig');    %����Ԫ�����ƽ�棨ֻ���ڵ���Ԫ�У�
mesh(ES,[60,30]);                     %����ά��״�棬�ӽǡ�60��30�� 
title('Error Surface Graph')
xlabel('W');
ylabel('B');
zlabel('Error')
figure(2)
[C,h] =contour(Wrange,Brange,ES,6);     %���ȸ���ͼ��ESΪ��
                                   %���صȸ��߾���C��������h���߻����ľ����
                                   %һ����һ���������Щ������CLABEL�����룬 
                                   %ÿ���������ÿ���ȸ��ߵĸ߶� 
clabel(C,h)                          %���ϸ߶�ֵ
colormap cool                        %��������ɫcool
hold on
plot(w1,b1,'r+')
xlabel('W');
ylabel('B');
hold off
figure(3)
plot(tr.perf)                           %���������
