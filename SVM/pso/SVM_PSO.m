%E:\OnlineDisk\OneDrive\OneDrive - Neuroinformatics Collaboratory\Github\Pattern Recognition\SVM
%% PSO for SVM parameter C & gamma
function gbest=SVM_PSO(tarin_data,train_label,parameter_dim,popsize,scale,maxstep)


%% 初始化粒子参数，公式里的常量
wmax=1.0;wmin=0.1;
c1=2;c2=2;
gbest_history=[];
pbest_history=[];
figure('NumberTitle', 'off', 'Name', 'PSO');
%% 常改动参数
%popsize=10;%population size
dim=parameter_dim;%几元函数，搜寻所在的超平面
maxstep=maxstep;%代数
%% 初值
x=zeros(popsize,dim);
for i=1:dim
    x(:,i)=rand(popsize,1).*(scale(i,2)-scale(i,1))/1.5;%初始化位置
    v(:,i)=rand(popsize,1).*(scale(i,2)-scale(i,1))/3;%初始化速度
end

pbest=x;
cost=fun_cost(x,scale,tarin_data,train_label);
[mincost,index]=min(cost);
gbest=pbest(index);
%% 循环
for gn_num=1:1:maxstep
    w=wmax-i*(wmax-wmin)/maxstep;
    r1=rand(popsize,dim);
    r2=rand(popsize,dim);
    %% 粒子开始运动
    x=x+v;
    x(x<0)=-x(x<0);
    %% 单个粒子最优位置
    newcost=fun_cost(x,scale,tarin_data,train_label);
    change=cost>newcost; %比较前后目标函数        
    
    pbest=(1-change).*pbest+change.*x;%更新各粒子最优位置
    cost=(1-change).*cost+change.*newcost;%更新损失函数   
    %% 粒子间通信最优位置
    [mincost,index]=min(cost);
    gbest=pbest(index,:);      
    %% 更新速度
    v=w*v+c1*r1.*(pbest-x)+c2*r2.*(gbest-x);    
    
    gbest_cost=fun_cost(gbest,scale,tarin_data,train_label)
    %% SVM寻优可视化
    
    pbest_history=[pbest_history;[pbest';cost']];
    gbest_history=[gbest_history;[gbest';gbest_cost]];
%      for i=1:1:popsize
%         %plot3(pbest_history(1:3:end-2,1),pbest_history(2:3:end-1,2),pbest_history(3:3:end,3));
%         %plot3(pbest_history(3*i-2,:)',pbest_history(3*i-1,:)',pbest_history(3*i,:)''-b');
%      end
    plot3(pbest_history(1:3:end-2,:)',pbest_history(2:3:end-1,:)',pbest_history(3:3:end,:)','-b');
    pause(0.05)
    hold on
    plot3(gbest_history(1:3:end-2,:)',gbest_history(2:3:end-1,:)',gbest_history(3:3:end,:)','-r');
    pause(0.05)
   %% 存图
   picname=[num2str(gn_num) '.fig'];%保存的文件名：如i=1时，picname=1.fig
   saveas(gcf,picname)
   pause(0.1)
    
end
%% display
gbest
gbest_cost=fun_cost(gbest,scale,tarin_data,train_label)
save('.\data\svm_pso.mat')