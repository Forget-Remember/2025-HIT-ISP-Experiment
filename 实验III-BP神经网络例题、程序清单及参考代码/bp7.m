% bp7;��4.7
%
P=[-3,2];T=[0.4,0.8];
max_epoch=100;                            %�����ѵ������ 
err1=[];xx=[];yy=[];                           %�������                     
lr=4;
W1=-4*rand(1);                              %��Ȩֵ��ƫ��ĳ�ֵ
b1=-4*rand(1);
x1=W1;y1=b1;
for i=1:max_epoch
    A1 = logsig(W1*P+b1*ones(1,2));           %�������
    E = T-A1;                               %�����
    D1 = A1.*(1-A1).*E;                      %�����ӦԪ�����
    dW1 = D1*P'*lr;                         %��Ȩֵ����
    db1 = D1*ones(2,1)*lr;                    %��ƫ������
    newx = W1(1,1) + dW1(1,1);               %�µ�Ȩֵ
    W1(1,1) = newx; xx =[xx newx];
    newy = b1(1)   + db1(1);                 %�µ�ƫ��  
    b1(1) = newy;   yy =[yy newy];                 
    SSE = sumsqr(E);                        %�����ƽ����
    err1=[err1   SSE];
    if (SSE<1e-3)
        break;
    end
end
Wrange=-4:0.2:4;Brange=-4:0.2:4;              %Wֵ����������Bֵ��������
ES=errsurf(P,T,Wrange,Brange,'logsig');       %����Ԫ�����ƽ�棨ֻ���ڵ���Ԫ�У�
[C,h] =contour(Wrange,Brange,ES);            %���ȸ���ͼ��ESΪ��
                                    %���صȸ��߾���C��������h���߻����ľ����
                                    %һ����һ���������Щ������CLABEL�����룬                                           %ÿ���������ÿ���ȸ��ߵĸ߶� 
clabel(C,h)                           %���ϸ߶�ֵ
colormap cool                         %��������ɫcool
axis('equal')
hold on
%plot(w1,b1,'r*')
plot(x1,y1,'r+')
plot([x1 xx],[y1 yy])                    %������仯����
hold off
figure(2)
plot(err1)                             %�����仯����
