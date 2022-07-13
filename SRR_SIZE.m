%% 清空环境
clc;clear all;close all;

%%  加载LSSVM训练好的预测模型
%load('modelCS.mat');
load('modelCO.mat');
%    lssvm相关参数
%    type='f';
%    kernel = 'RBF_kernel';
%    proprecess='proprecess';
%    for CS gam=1000;
%    for CS sig2=5.4403;
%    for CO gam=993.731;
%    for CO sig2=4.1252;

%% 目标函数
%    下列函数为：
%    y=(D3+D2+66.4)*W*H
%    约束条件为：
%      Xtest = [D3 D2 W H];
%     [train_predict_yy,zt,model] = simlssvm(model,Xtest);
%     for Cs 8.25*1.05=8.6625      train_predict_yy<=8.6625
%     for Co 30.901*1.05=32.44605  train_predict_yy<=32.44605
a=[1,1,0,0];
b=[0,0,1,0];
c=[0,0,0,1];
fun= @(X)(a*X+66.4)*(b*X)*(c*X); 

%% 设置种群参数
%   需要自行配置
sizepop = 50;                    % 初始种群个数
dim = 4;                         % 空间维数
ger = 500;                       % 最大迭代次数    
xlimit_max = [20;100;300;300];   % 设置位置参数限制(矩阵的形式可以多维)D3 D2 W H
xlimit_min = [12.5;8;40;40];
vlimit_max = 0.5*xlimit_max;     % 设置速度限制
vlimit_min = -0.5*xlimit_min;
c_1 = 0.8;                       % 惯性权重
c_2 = 0.5;                       % 自我学习因子
c_3 = 0.5;                       % 群体学习因子 

%% 生成初始种群
%  首先随机生成初始种群位置
%  然后随机生成初始种群速度
%  然后初始化个体历史最佳位置，以及个体历史最佳适应度
%  然后初始化群体历史最佳位置，以及群体历史最佳适应度
for i=1:dim
    for j=1:sizepop
        pop_x(i,j) = xlimit_min(i)+(xlimit_max(i) - xlimit_min(i))*rand;  % 初始种群的位置
        pop_v(i,j) = vlimit_min(i)+(vlimit_max(i) - vlimit_min(i))*rand;  % 初始种群的速度
    end
end                 
gbest = pop_x;                                % 每个个体的历史最佳位置
for j=1:sizepop
    Xtest = pop_x(:,j)';
    [train_predict_yy,zt,model] = simlssvm(model,Xtest);
    if train_predict_yy<=32.3114
        fitness_gbest(j) = fun(pop_x(:,j));                      % 每个个体的历史最佳适应度
    else
        fitness_gbest(j) = 10^8;             % 惩罚因子
    end
end                  
zbest = pop_x(:,1);                           % 种群的历史最佳位置
fitness_zbest = fitness_gbest(1);             % 种群的历史最佳适应度
for j=1:sizepop
    if fitness_gbest(j) < fitness_zbest       % 如果求最小值，则为<; 如果求最大值，则为>; 
        zbest = pop_x(:,j);
        fitness_zbest=fitness_gbest(j);
    end
end


%% 粒子群迭代
%    更新速度并对速度进行边界处理    
%    更新位置并对位置进行边界处理
%    进行自适应变异
%    进行约束条件判断并计算新种群各个个体位置的适应度
%    新适应度与个体历史最佳适应度做比较
%    个体历史最佳适应度与种群历史最佳适应度做比较
%    再次循环或结束

iter = 1;                        %迭代次数
record = zeros(ger, 1);          % 记录器
while iter <= ger
    for j=1:sizepop
        %    更新速度并对速度进行边界处理 
        pop_v(:,j)= c_1 * pop_v(:,j) + c_2*rand*(gbest(:,j)-pop_x(:,j))+c_3*rand*(zbest-pop_x(:,j));% 速度更新
        for i=1:dim
            if  pop_v(i,j) > vlimit_max(i)
                pop_v(i,j) = vlimit_max(i);
            end
            if  pop_v(i,j) < vlimit_min(i)
                pop_v(i,j) = vlimit_min(i);
            end
        end
        
        %    更新位置并对位置进行边界处理
        pop_x(:,j) = pop_x(:,j) + pop_v(:,j);% 位置更新
        for i=1:dim
            if  pop_x(i,j) > xlimit_max(i)
                pop_x(i,j) = xlimit_max(i);
            end
            if  pop_x(i,j) < xlimit_min(i)
                pop_x(i,j) = xlimit_min(i);
            end
        end
        
        %    进行自适应变异
        if rand > 0.85
            i=ceil(dim*rand);
            pop_x(i,j)=xlimit_min(i) + (xlimit_max(i) - xlimit_min(i)) * rand;
        end
  
        %    进行约束条件判断并计算新种群各个个体位置的适应度
        Xtest = pop_x(:,j)';
        [train_predict_yy,zt,model] = simlssvm(model,Xtest);
            if train_predict_yy<=32.3114
                fitness_pop(j) = fun(pop_x(:,j));                      % 每个个体的历史最佳适应度
            else
                fitness_pop(j) = 10^8;
            end

        %    新适应度与个体历史最佳适应度做比较
        if fitness_pop(j) < fitness_gbest(j)       % 如果求最小值，则为<; 如果求最大值，则为>; 
            gbest(:,j) = pop_x(:,j);               % 更新个体历史最佳位置            
            fitness_gbest(j) = fitness_pop(j);     % 更新个体历史最佳适应度
        end   
        
        %    个体历史最佳适应度与种群历史最佳适应度做比较
        if fitness_gbest(j) < fitness_zbest        % 如果求最小值，则为<; 如果求最大值，则为>; 
            zbest = gbest(:,j);                    % 更新群体历史最佳位置  
            fitness_zbest=fitness_gbest(j);        % 更新群体历史最佳适应度  
        end    
    end
    
    record(iter) = fitness_zbest;%最大值记录
    
    iter = iter+1;

end
%% 迭代结果输出
plot(record);title('收敛过程')
disp(['最优值：',num2str(fitness_zbest)]);
disp('变量取值：');
disp(zbest);
