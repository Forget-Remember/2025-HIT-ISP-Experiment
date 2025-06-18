%bp5;��4.5
%
time=[];err=[];epo=[];

for i=1:6                                   %�ֱ����ȡS1= 1 - 6�ڵ�ʱ������ѵ��
t(i)=cputime;                                      %��ʼCPUʱ��
P=[0 0 1 1;0 1 0 1];
T=[0 1 1 0];
net=newcf(minmax(P),[i,1],{'tansig' 'purelin'},'traingd' ); 
                                                 %��������ǰ��BP����
net.trainParam.show = 50;                            %ÿ50����ʾһ�ν��
net.trainParam.lr = 0.05;                           %ѧϰ����
net.trainParam.epochs = 300;                       %���ѭ������
[net,tr]=train(net,P,T);                            %ѵ������
Y=sim(net,P);                                   %����������
err=[err;tr.perf];
time(i)=cputime-t(i);                             %�����������ʱ��
end

plot(err(1,:));                                   %��������6�������ѵ�����ͼ    
hold on
plot(err(2,:),'g:');
plot(err(3,:),'g-');
plot(err(4,:),':');
plot(err(5,:),'r:');
plot(err(6,:),'r-');
hold off
time
