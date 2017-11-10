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
%% 预处理
%归一化
train_data=train_data./(max(max(abs(train_data)),max(abs(test_data)))-min(min(abs(train_data)),min(abs(test_data))));
test_data=test_data./(max(max(abs(train_data)),max(abs(test_data)))-min(min(abs(train_data)),min(abs(test_data))));
%% 寻优参数
parameter_dim=2;
cmin=2^(-5);
cmax=2^15;
gamma_min=2^(-15);
gamma_max=2^3;
scale=[cmin,cmax;gamma_min,gamma_max];
popsize=20;
maxstep=10;
%fold_cv=5;
%cstep=10;
%gstep=10;
%accstep=1;
%% 
gbest=SVM_PSO(train_data,train_label,parameter_dim,popsize,scale,maxstep);
c_best=gbest(1);
gamma_best=gbest(2);
cmd= [' -c ',num2str(c_best),' -g ',num2str(gamma_best)];
model=svmtrain(train_label,train_data,cmd);
%model=svmtrain(train_label,train_data,'-t 2 -v 5');
%model=svmtrain(train_label,train_data,'-t 2 -v 18');
%load svm_pso.mat
[predict_label] = svmpredict(test_label,test_data,model);

