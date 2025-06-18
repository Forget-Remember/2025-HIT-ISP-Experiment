% bp10;Àı4.10
%
P=-1:0.1:1;
T=[-0.96 -0.577 -0.0729 0.377 0.641 0.66 0.461 0.1336 ...
   -0.201 -0.434 -0.5 -0.393 -0.1647 0.0988 0.3072 ...
   0.396 0.3449 0.1816 -0.0312 -0.2183 -0.3201];

net=newff(minmax(P),[5,1],{'tansig','purelin'},'traingda');
net.trainParam.show = 50;
net.trainParam.lr = 0.05;
net.trainParam.lr_inc = 1.08;
net.trainParam.lr_dec = 0.6;  
net.trainParam.epochs = 2000;
net.trainParam.goal = 9.5238e-004; %  sse=0.02

[net,tr]=train(net,P,T);
plot(tr.lr);
figure(2)
plot(tr.perf);
