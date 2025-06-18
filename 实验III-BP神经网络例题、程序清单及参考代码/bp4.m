%bp4;例4.4
%
P=[-3,2];T=[0.4,0.8];
Wrange=-4:0.4:4;Brange=-4:0.4:4;            %W值的行向量、B值的行向量
ES=errsurf(P,T,Wrange,Brange,'logsig');       %求单神经元的误差平面（只用在单神经元中）
mesh(ES,[60,30]);                         %作三维网状面，视角【60，30】 
title('Error Surface Graph')                  %标题
xlabel('W')
ylabel('B')
zlabel('Sum Squared Error')

max_epoch=1500;                         %赋最大训练次数 
err=[];xx=[];yy=[];                         %定义矩阵                     
lr=0.5;
W1=-4*rand(1);                           %赋权值和偏差的初值
b1=-4*rand(1);
x1=W1;y1=b1;
for i=1:max_epoch
    A1 = logsig(W1*P+b1*ones(1,2));        %计算输出
    E = T-A1;                            %求误差
    D1 = A1.*(1-A1).*E;                   %矩阵对应元素相乘
    dW1 = D1*P'*lr;                      %求权值增量
    db1 = D1*ones(2,1)*lr;                 %求偏差增量
    newx = W1(1,1) + dW1(1,1);            %新的权值
    W1(1,1) = newx; xx =[xx newx];
    newy = b1(1)   + db1(1);              %新的偏差  
    b1(1) = newy;   yy =[yy newy];                 
    SSE = sumsqr(E);                  %求误差平方和
    err=[err SSE];
    if (SSE<1e-8)
        break;
    end
end

figure(2)
[C,h] =contour(Wrange,Brange,ES);       %作等高线图，ES为高
                                    %返回等高线矩阵C，列向量h是线或对象的句柄，
                                    %一条线一个句柄，这些被用作CLABEL的输入， 
                                    %每个对象包含每个等高线的高度 
clabel(C,h)                           %标上高度值
colormap cool                         %背景的颜色cool
axis('equal')
hold on
plot(x1,y1,'r+')
plot(xx,yy)                            %作矩阵变化曲线
hold off

figure(3)
plot(err)                              %作误差曲线
