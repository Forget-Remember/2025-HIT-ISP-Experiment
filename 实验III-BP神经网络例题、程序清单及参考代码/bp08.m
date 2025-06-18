%�۲�Ȩֵ��ƫ��ı仯�г������£�
%bp08;��4.8
%
P=[-6 -6.1 -4.1 -4 4 4.1 6 6.1];
T=[0.0 0.0 0.97 0.99 0.01 0.03 1 1];
xx=[];yy=[];err=[];
W1=-0.0511;                              %��Ȩֵ��ƫ��ĳ�ֵ
b1=-2.4462;
x1=W1;y1=b1;
lr=0.1
max_epoch=2000;
for i=1:max_epoch
    A1 = logsig(W1*P+b1*ones(1,8));         %�������
    E = T-A1;                             %�����
    D1 = A1.*(1-A1).*E;                    %�����ӦԪ�����
    dW1 = D1*P'*lr;                       %��Ȩֵ����
    db1 = D1*ones(8,1)*lr;                  %��ƫ������
    newx = W1(1,1) + dW1(1,1);             %�µ�Ȩֵ
    W1(1,1) = newx; xx =[xx newx];
    newy = b1(1)   + db1(1);               %�µ�ƫ��  
    b1(1) = newy;   yy =[yy newy];                
    SSE = sumsqr(E);                      %�����ƽ����
    err=[err   SSE];
    if (SSE<0.2)
        break;
    end
end
Wrange=-1:0.1:1;Brange=-3:0.2:2;            %Wֵ����������Bֵ��������
ES=errsurf(P,T,Wrange,Brange,'logsig');       %����Ԫ�����ƽ�棨ֻ���ڵ���Ԫ�У�
[C,h] =contour(Wrange,Brange,ES,6);      %���ȸ���ͼ��ESΪ��
                                    %���صȸ��߾���C��������h���߻����ľ����
                                     %һ����һ���������Щ������CLABEL�����룬 
                                     %ÿ���������ÿ���ȸ��ߵĸ߶� 
clabel(C,h)                            %���ϸ߶�ֵ
colormap cool                         %��������ɫcool
hold on
%plot(w1,b1,'r*')
plot(x1,y1,'r+')
plot([x1 xx],[y1 yy])                    %��Ȩֵ��ƫ��仯����
hold off
figure(2)
plot(err)
