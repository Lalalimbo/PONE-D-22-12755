%% ��ջ���
clc;clear all;close all;

%%  ����LSSVMѵ���õ�Ԥ��ģ��
%load('modelCS.mat');
load('modelCO.mat');
%    lssvm��ز���
%    type='f';
%    kernel = 'RBF_kernel';
%    proprecess='proprecess';
%    for CS gam=1000;
%    for CS sig2=5.4403;
%    for CO gam=993.731;
%    for CO sig2=4.1252;

%% Ŀ�꺯��
%    ���к���Ϊ��
%    y=(D3+D2+66.4)*W*H
%    Լ������Ϊ��
%      Xtest = [D3 D2 W H];
%     [train_predict_yy,zt,model] = simlssvm(model,Xtest);
%     for Cs 8.25*1.05=8.6625      train_predict_yy<=8.6625
%     for Co 30.901*1.05=32.44605  train_predict_yy<=32.44605
a=[1,1,0,0];
b=[0,0,1,0];
c=[0,0,0,1];
fun= @(X)(a*X+66.4)*(b*X)*(c*X); 

%% ������Ⱥ����
%   ��Ҫ��������
sizepop = 50;                    % ��ʼ��Ⱥ����
dim = 4;                         % �ռ�ά��
ger = 500;                       % ����������    
xlimit_max = [20;100;300;300];   % ����λ�ò�������(�������ʽ���Զ�ά)D3 D2 W H
xlimit_min = [12.5;8;40;40];
vlimit_max = 0.5*xlimit_max;     % �����ٶ�����
vlimit_min = -0.5*xlimit_min;
c_1 = 0.8;                       % ����Ȩ��
c_2 = 0.5;                       % ����ѧϰ����
c_3 = 0.5;                       % Ⱥ��ѧϰ���� 

%% ���ɳ�ʼ��Ⱥ
%  ����������ɳ�ʼ��Ⱥλ��
%  Ȼ��������ɳ�ʼ��Ⱥ�ٶ�
%  Ȼ���ʼ��������ʷ���λ�ã��Լ�������ʷ�����Ӧ��
%  Ȼ���ʼ��Ⱥ����ʷ���λ�ã��Լ�Ⱥ����ʷ�����Ӧ��
for i=1:dim
    for j=1:sizepop
        pop_x(i,j) = xlimit_min(i)+(xlimit_max(i) - xlimit_min(i))*rand;  % ��ʼ��Ⱥ��λ��
        pop_v(i,j) = vlimit_min(i)+(vlimit_max(i) - vlimit_min(i))*rand;  % ��ʼ��Ⱥ���ٶ�
    end
end                 
gbest = pop_x;                                % ÿ���������ʷ���λ��
for j=1:sizepop
    Xtest = pop_x(:,j)';
    [train_predict_yy,zt,model] = simlssvm(model,Xtest);
    if train_predict_yy<=32.3114
        fitness_gbest(j) = fun(pop_x(:,j));                      % ÿ���������ʷ�����Ӧ��
    else
        fitness_gbest(j) = 10^8;             % �ͷ�����
    end
end                  
zbest = pop_x(:,1);                           % ��Ⱥ����ʷ���λ��
fitness_zbest = fitness_gbest(1);             % ��Ⱥ����ʷ�����Ӧ��
for j=1:sizepop
    if fitness_gbest(j) < fitness_zbest       % �������Сֵ����Ϊ<; ��������ֵ����Ϊ>; 
        zbest = pop_x(:,j);
        fitness_zbest=fitness_gbest(j);
    end
end


%% ����Ⱥ����
%    �����ٶȲ����ٶȽ��б߽紦��    
%    ����λ�ò���λ�ý��б߽紦��
%    ��������Ӧ����
%    ����Լ�������жϲ���������Ⱥ��������λ�õ���Ӧ��
%    ����Ӧ���������ʷ�����Ӧ�����Ƚ�
%    ������ʷ�����Ӧ������Ⱥ��ʷ�����Ӧ�����Ƚ�
%    �ٴ�ѭ�������

iter = 1;                        %��������
record = zeros(ger, 1);          % ��¼��
while iter <= ger
    for j=1:sizepop
        %    �����ٶȲ����ٶȽ��б߽紦�� 
        pop_v(:,j)= c_1 * pop_v(:,j) + c_2*rand*(gbest(:,j)-pop_x(:,j))+c_3*rand*(zbest-pop_x(:,j));% �ٶȸ���
        for i=1:dim
            if  pop_v(i,j) > vlimit_max(i)
                pop_v(i,j) = vlimit_max(i);
            end
            if  pop_v(i,j) < vlimit_min(i)
                pop_v(i,j) = vlimit_min(i);
            end
        end
        
        %    ����λ�ò���λ�ý��б߽紦��
        pop_x(:,j) = pop_x(:,j) + pop_v(:,j);% λ�ø���
        for i=1:dim
            if  pop_x(i,j) > xlimit_max(i)
                pop_x(i,j) = xlimit_max(i);
            end
            if  pop_x(i,j) < xlimit_min(i)
                pop_x(i,j) = xlimit_min(i);
            end
        end
        
        %    ��������Ӧ����
        if rand > 0.85
            i=ceil(dim*rand);
            pop_x(i,j)=xlimit_min(i) + (xlimit_max(i) - xlimit_min(i)) * rand;
        end
  
        %    ����Լ�������жϲ���������Ⱥ��������λ�õ���Ӧ��
        Xtest = pop_x(:,j)';
        [train_predict_yy,zt,model] = simlssvm(model,Xtest);
            if train_predict_yy<=32.3114
                fitness_pop(j) = fun(pop_x(:,j));                      % ÿ���������ʷ�����Ӧ��
            else
                fitness_pop(j) = 10^8;
            end

        %    ����Ӧ���������ʷ�����Ӧ�����Ƚ�
        if fitness_pop(j) < fitness_gbest(j)       % �������Сֵ����Ϊ<; ��������ֵ����Ϊ>; 
            gbest(:,j) = pop_x(:,j);               % ���¸�����ʷ���λ��            
            fitness_gbest(j) = fitness_pop(j);     % ���¸�����ʷ�����Ӧ��
        end   
        
        %    ������ʷ�����Ӧ������Ⱥ��ʷ�����Ӧ�����Ƚ�
        if fitness_gbest(j) < fitness_zbest        % �������Сֵ����Ϊ<; ��������ֵ����Ϊ>; 
            zbest = gbest(:,j);                    % ����Ⱥ����ʷ���λ��  
            fitness_zbest=fitness_gbest(j);        % ����Ⱥ����ʷ�����Ӧ��  
        end    
    end
    
    record(iter) = fitness_zbest;%���ֵ��¼
    
    iter = iter+1;

end
%% ����������
plot(record);title('��������')
disp(['����ֵ��',num2str(fitness_zbest)]);
disp('����ȡֵ��');
disp(zbest);
