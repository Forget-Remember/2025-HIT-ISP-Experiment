%观察权值和偏差的变化有程序如下：
%bp08;例4.8
%
P=[-6 -6.1 -4.1 -4 4 4.1 6 6.1];
T=[0.0 0.0 0.97 0.99 0.01 0.03 1 1];
xx=[];yy=[];err=[];
W1=-0.0511;                              %赋权值和偏差的初值
b1=-2.4462;
x1=W1;y1=b1;
lr=0.1
max_epoch=2000;
for i=1:max_epoch
    A1 = logsig(W1*P+b1*ones(1,8));         %计算输出
    E = T-A1;                             %求误差
    D1 = A1.*(1-A1).*E;                    %矩阵对应元素相乘
    dW1 = D1*P'*lr;                       %求权值增量
    db1 = D1*ones(8,1)*lr;                  %求偏差增量
    newx = W1(1,1) + dW1(1,1);             %新的权值
    W1(1,1) = newx; xx =[xx newx];
    newy = b1(1)   + db1(1);               %新的偏差  
    b1(1) = newy;   yy =[yy newy];                
    SSE = sumsqr(E);                      %求误差平方和
    err=[err   SSE];
    if (SSE<0.2)
        break;
    end
end
Wrange=-1:0.1:1;Brange=-3:0.2:2;            %W值的行向量、B值的行向量
ES=errsurf(P,T,Wrange,Brange,'logsig');       %求单神经元的误差平面（只用在单神经元中）
[C,h] =contour(Wrange,Brange,ES,6);      %作等高线图，ES为高
                                    %返回等高线矩阵C，列向量h是线或对象的句柄，
                                     %一条线一个句柄，这些被用作CLABEL的输入， 
                                     %每个对象包含每个等高线的高度 
clabel(C,h)                            %标上高度值
colormap cool                         %背景的颜色cool
hold on
%plot(w1,b1,'r*')
plot(x1,y1,'r+')
plot([x1 xx],[y1 yy])                    %作权值和偏差变化曲线
hold off
figure(2)
plot(err)
