%E:\OnlineDisk\OneDrive\OneDrive - Neuroinformatics Collaboratory\Github\Pattern Recognition\SVM
%% ·��
file_path=mfilename('fullpath');
i=strfind(file_path,'\');
file_path=file_path(1:i(end));
cd(file_path);
addpath('.\pso') ;
%% ��ȡ����
clear;clc;close all;
t0=cputime;
%��train&test data
train_data=load('.\data\train_data.mat');
train_label=load('.\data\train_label.mat');
test_data=load('.\data\test_data.mat');
test_label=load('.\data\test_label.mat');
train_data=train_data.train_data;
train_label=train_label.train_label;
test_data=test_data.test_data;
test_label=test_label.test_label;
%% Ѱ�Ų���
parameter_dim=2;
cmin=2^(-5);
cmax=2^15;
gamma_min=2^(-15);
gamma_max=2^3;
scale=[cmin,cmax;gamma_min,gamma_max];
popsize=20;
maxstep=20;
%% Ѱ��
gbest=SVM_PSO(train_data,train_label,parameter_dim,popsize,scale,maxstep);
c_best=gbest(1);
gamma_best=gbest(2);
%% ѵ��
cmd= ['-t 2 -c ',num2str(c_best),' -g ',num2str(gamma_best)];
model=svmtrain(train_label,train_data,cmd);
%% ����
[predicted_label, accuracy, decision_values] = svmpredict(test_label,test_data,model);
rmpath .\pso