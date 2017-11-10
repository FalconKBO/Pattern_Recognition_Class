%E:\OnlineDisk\OneDrive\OneDrive - Neuroinformatics Collaboratory\Github\Pattern Recognition\SVM
%% 路径
file_path=mfilename('fullpath');
i=strfind(file_path,'\');
file_path=file_path(1:i(end));
cd(file_path);
addpath('.\pso') ;
%% 读取数据
clear;clc;close all;
t0=cputime;
%读train&test data
train_data=load('.\own_data\train_data.mat');
train_label=load('.\own_data\train_label.mat');
test_data=load('.\own_data\test_data.mat');
test_label=load('.\own_data\test_label.mat');
train_data=train_data.train_data;
train_label=train_label.train_label;
test_data=test_data.test_data;
test_label=test_label.test_label;
%% 预处理;
%归一化
alpha=(max(max(abs(train_data)),max(abs(test_data)))-min(min(abs(train_data)),min(abs(test_data))));
train_data=train_data./alpha;
test_data=test_data./alpha;
figure('NumberTitle', 'off', 'Name', 'svm for sin data');
%% train & predict
%pso优化后的参数，找的c有点大[38644.5518185840,0.221398585898100]
    c_best=100.5518185840;
    gamma_best=0.221398585898100;
% for gn_num=1:1:40
%     gn_num
%     %c_best=0.5^(21-gn_num);
%     gamma_best=0.5^(21-gn_num);
    
    

    cmd= ['-t 2 -c ',num2str(c_best),' -g ',num2str(gamma_best)];
    %训练
    model=svmtrain(train_label,train_data,cmd);
    save .\own_data\fun_svm_predict.mat model
    %训练集结果
    cmd=strcat('-v 5 ',cmd);
    acc=svmtrain(train_label,train_data,cmd);
    %测试
    [predicted_label, accuracy, decision_values] = svmpredict(test_label,test_data,model);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% 小范围内数据的分类器输出分布
    x=[-1.5:3/99:1.5]';
    y=[-1.5:3/99:1.5]';
    [X,Y]=meshgrid(x,y);
    X=reshape(X,[size(x).^2,1]);
    Y=reshape(Y,[size(x).^2,1]);
    Z=fun_svm_predict([X,Y]);
    X1=reshape(X,[size(x,1),size(x,1)]);
    Y1=reshape(Y,[size(y,1),size(y,1)]);
    Z1=reshape(Z,[size(x,1),size(y,1)]);
    subplot(2,2,1);
    surfc(X1,Y1,Z1,'EdgeColor','none');
    hold on
    contour3(X1,Y1,Z1,'Color','g');
    xlabel('X');ylabel('Y');zlabel('Z');title('核SVM输出');
    hold off
    %% 一定范围内数据的分类器输出分布
    x=[-6:12/99:6]';
    y=[-6:12/99:6]';
    [X,Y]=meshgrid(x,y);
    X=reshape(X,[size(x).^2,1]);
    Y=reshape(Y,[size(x).^2,1]);
    Z=fun_svm_predict([X,Y]);
    X1=reshape(X,[size(x,1),size(x,1)]);
    Y1=reshape(Y,[size(y,1),size(y,1)]);
    Z1=reshape(Z,[size(x,1),size(y,1)]);
    subplot(2,2,2);
    surfc(X1,Y1,Z1,'EdgeColor','none');
    hold on
    contour3(X1,Y1,Z1,'Color','g');
    xlabel('X');ylabel('Y');zlabel('Z');title('核SVM输出');
    hold off
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% draw result 
    inlayer=test_data(test_label==1,1:end);
    label_true{1}=test_label(test_label==1);
    outlayer=test_data(test_label==-1,1:end);
    label_true{2}=test_label(test_label==-1);
    subplot(2,2,3);    
    plot(inlayer(:,1),inlayer(:,2),'.r')
    hold on
    plot(outlayer(:,1),outlayer(:,2),'.b')
    hold on
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% draw suport vector
    sv_data{1}=train_data(model.sv_indices(1:model.nSV),:);
    sv_data{2}=train_data(model.sv_indices(model.nSV+1:end),:);
    sv=full(model.SVs);
    plot(sv_data{1}(:,1),sv_data{1}(:,2),'or')
    hold on
    plot(sv_data{2}(:,1),sv_data{2}(:,2),'ob')
    r=sum((sv_data{2}(:,1).^2+sv_data{2}(:,2).^2).^(1/2))/size(sv_data{2},1);
    %
    inlayer=train_data(train_label==1,1:end);
    label_true{1}=train_label(train_label==1);
    outlayer=train_data(train_label==-1,1:end);
    label_true{2}=train_label(train_label==-1);
    %% draw edge
    [C,h] = contour(X1, Y1, Z1,-1:1:1);
    clabel(C,h,'Color','r');
    grid;
    xlabel('X');ylabel('Y');title('非线性分界面和支持向量产生的边界');
    hold off
    %% draw ROC
    subplot(2,2,4);
    [totalScores,index]  = sort(decision_values);
    labels = test_label(index);
    truePositive = zeros(1,length(totalScores)+1);
    trueNegative = zeros(1,length(totalScores)+1);
    falsePositive = zeros(1,length(totalScores)+1);
    falseNegative = zeros(1,length(totalScores)+1);
    for i = 1:length(totalScores)
        if labels(i) == 1
            truePositive(1) = truePositive(1)+1;
        else
            falsePositive(1) = falsePositive(1) +1;
        end
    end
    for i = 1:length(totalScores)
       if labels(i) == 1
           truePositive(i+1) = truePositive(i)-1;
           falsePositive(i+1) = falsePositive(i);
       else
           falsePositive(i+1) = falsePositive(i)-1;
           truePositive(i+1) = truePositive(i);
       end
    end
    truePositive = truePositive/truePositive(1);
    falsePositive = falsePositive/falsePositive(1);
    plot(falsePositive,truePositive);
    xlabel('falsePositive');ylabel('truePositive');title('Receiver Operating Characteristic Curve');
    hold off
%    %% 存图
%    picname=[num2str(gn_num) '.fig'];%保存的文件名：如i=1时，picname=1.fig
%    saveas(gcf,picname)
%    pause(0.1)
% 
% end


rmpath .\pso