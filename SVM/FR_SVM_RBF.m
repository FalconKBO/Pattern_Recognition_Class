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
train_data=load('.\data\train_data.mat');
train_label=load('.\data\train_label.mat');
test_data=load('.\data\test_data.mat');
test_label=load('.\data\test_label.mat');
train_data=train_data.train_data;
train_label=train_label.train_label;
test_data=test_data.test_data;
test_label=test_label.test_label;
%% 
%pso优化后的参数，容错能力不是很好，所以有些过拟合，c设成0.287好像在测试集表现更好
%[5309.28751114026,0.133464099052983][7247.18109651587,0.170471352601388][4.5948,0.125]c=0.8,g=0.009[15854.9019173380,0.246347253996144]
c_best=15854.9019173380;
gamma_best=0.246347253996144;
%% 训练
cmd= ['-t 2 -c ',num2str(c_best),' -g ',num2str(gamma_best)];
model=svmtrain(train_label,train_data,cmd);
cmd=strcat('-v 5 ',cmd);
acc=svmtrain(train_label,train_data,cmd);
[predicted_label, accuracy, decision_values] = svmpredict(test_label,test_data,model);
%%
face=test_data(predicted_label==1,1:end);
label_true{1}=test_label(predicted_label==1);
nonface=test_data(predicted_label==-1,1:end);
label_true{2}=test_label(predicted_label==-1);
%%
for i=1:size(face,1)
    img_face{i}=reshape(uint8(face(i,:)*255),[19,19]);
end
for i=1:size(nonface,1)
    img_nonface{i}=reshape(uint8(nonface(i,:)*255),[19,19]);
end
for i=1:size(train_data,1)
    img_train{i}=reshape(uint8(train_data(i,:)*255),[19,19]);
end
%%
%随机显示分类结果中部分图像及其结果
label_mean={'人脸';'非人脸'};
figure('NumberTitle', 'off', 'Name', '人脸');
perm = randperm(size(img_face,2),20);
for i = 1:20
    subplot(4,5,i);
    imshow(img_face{perm(i)});  
    title(strcat('real: ',label_mean{(label_true{1}(perm(i))==-1)+1}));    
end
suptitle('人脸');
figure('NumberTitle', 'off', 'Name', '非人脸');
perm = randperm(size(img_nonface,2),20);
for i = 1:20
    subplot(4,5,i);
    imshow(img_nonface{perm(i)}); 
    title(strcat('real: ',label_mean{(label_true{2}(perm(i))==-1)+1}));    
end
suptitle('非人脸');
%% draw ROC
% figure('NumberTitle', 'off', 'Name', 'ROC');
% [totalScores,index]  = sort(decision_values);
% labels = test_label(index);
% truePositive = zeros(1,length(totalScores)+1);
% trueNegative = zeros(1,length(totalScores)+1);
% falsePositive = zeros(1,length(totalScores)+1);
% falseNegative = zeros(1,length(totalScores)+1);
% for i = 1:length(totalScores)
%     if labels(i) == 1
%         truePositive(1) = truePositive(1)+1;
%     else
%         falsePositive(1) = falsePositive(1) +1;
%     end
% end
% 
% for i = 1:length(totalScores)
%    if labels(i) == 1
%        truePositive(i+1) = truePositive(i)-1;
%        falsePositive(i+1) = falsePositive(i);
%    else
%        falsePositive(i+1) = falsePositive(i)-1;
%        truePositive(i+1) = truePositive(i);
%    end
% end
% truePositive = truePositive/truePositive(1);
% falsePositive = falsePositive/falsePositive(1);
% 
% plot(falsePositive,truePositive);

[X,Y,T,AUC] =perfcurve(test_label,decision_values,'1');
figure('NumberTitle', 'off', 'Name', 'ROC & PR'); 
subplot(1,2,1);
plot(X,Y)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification')
%由于像素值的分布符合多维正态分布，所以贝叶斯决策和EM效果很好
text(0.6,0.4,['AUC= ',num2str(AUC)]);

[~,best_threshold_index]=max(1-X+Y);
best_threshhold=T(best_threshold_index);
%画PR曲线
[X,Y] =perfcurve(test_label,decision_values,'1','xCrit', 'reca', 'yCrit', 'prec');
subplot(1,2,2);
plot(X,Y)
xlabel('Recall') 
ylabel('Precision')
title('PRC for Classification')
%正确率
%acc=sum(predicted_label==test_label)/size(test_label,1)






consumed_time=cputime-t0;
rmpath .\pso
