%E:\OnlineDisk\OneDrive\OneDrive - Neuroinformatics Collaboratory\Github\Pattern Recognition\SVM
%% PSO for SVM parameter C & gamma
function gbest=SVM_PSO(tarin_data,train_label,parameter_dim,popsize,scale,maxstep)


%% ��ʼ�����Ӳ�������ʽ��ĳ���
wmax=1.0;wmin=0.1;
c1=2;c2=2;
gbest_history=[];
pbest_history=[];
figure('NumberTitle', 'off', 'Name', 'PSO');
%% ���Ķ�����
%popsize=10;%population size
dim=parameter_dim;%��Ԫ��������Ѱ���ڵĳ�ƽ��
maxstep=maxstep;%����
%% ��ֵ
x=zeros(popsize,dim);
for i=1:dim
    x(:,i)=rand(popsize,1).*(scale(i,2)-scale(i,1))/1.5;%��ʼ��λ��
    v(:,i)=rand(popsize,1).*(scale(i,2)-scale(i,1))/3;%��ʼ���ٶ�
end

pbest=x;
cost=fun_cost(x,scale,tarin_data,train_label);
[mincost,index]=min(cost);
gbest=pbest(index);
%% ѭ��
for gn_num=1:1:maxstep
    w=wmax-i*(wmax-wmin)/maxstep;
    r1=rand(popsize,dim);
    r2=rand(popsize,dim);
    %% ���ӿ�ʼ�˶�
    x=x+v;
    x(x<0)=-x(x<0);
    %% ������������λ��
    newcost=fun_cost(x,scale,tarin_data,train_label);
    change=cost>newcost; %�Ƚ�ǰ��Ŀ�꺯��        
    
    pbest=(1-change).*pbest+change.*x;%���¸���������λ��
    cost=(1-change).*cost+change.*newcost;%������ʧ����   
    %% ���Ӽ�ͨ������λ��
    [mincost,index]=min(cost);
    gbest=pbest(index,:);      
    %% �����ٶ�
    v=w*v+c1*r1.*(pbest-x)+c2*r2.*(gbest-x);    
    
    gbest_cost=fun_cost(gbest,scale,tarin_data,train_label)
    %% SVMѰ�ſ��ӻ�
    
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
   %% ��ͼ
   picname=[num2str(gn_num) '.fig'];%������ļ�������i=1ʱ��picname=1.fig
   saveas(gcf,picname)
   pause(0.1)
    
end
%% display
gbest
gbest_cost=fun_cost(gbest,scale,tarin_data,train_label)
save('.\data\svm_pso.mat')