%E:\OnlineDisk\OneDrive\OneDrive - Neuroinformatics Collaboratory\Github\Pattern Recognition\image segmentation
%% 读取图片及mask
%% 读取训练集并画出分布
%% 根据训练集计算两类分布的参数
%% 根据训练集计算先验概率
%% 计算新样本的条件概率
%% 最大化后验概率分类
%% 显示

%% 性能评估
%%读取数据
clear;clc;close all;
t0=cputime;
file_path=mfilename('fullpath');
i=strfind(file_path,'\');
file_path=file_path(1:i(end));
cd(file_path);
img=imread('.\data\309.bmp');
mask_file=load('.\data\Mask.mat','-mat','Mask');
mask=mask_file.Mask;
img_rgb_masked=img;
img_rgb_masked(:,:,1)=double(img(:,:,1)).*mask;
img_rgb_masked(:,:,2)=double(img(:,:,2)).*mask;
img_rgb_masked(:,:,3)=double(img(:,:,3)).*mask;
figure('NumberTitle', 'off', 'Name', '按rgb分割nemo'); 
subplot(2,3,1);
imshow(img_rgb_masked);
title('原来的nemo');
%% 读取并画出训练集分布
train_set_file=load('.\data\array_sample.mat');
train_set=train_set_file.array_sample;
white_set=train_set(train_set(:,5)==-1,2:4);
red_set=train_set(train_set(:,5)==1,2:4);
%% RGB三者构成了三维的正态分布,画出在三维RGB空间的散点图
%scatter3(train_set(:,1),train_set(:,2),train_set(:,3),'.');
subplot(2,3,3);
scatter3(white_set(:,1),white_set(:,2),white_set(:,3),'.b');
hold on
scatter3(red_set(:,1),red_set(:,2),red_set(:,3),'.r');
title('训练集数据')
%% 计算两类分布的参数，两个三维高斯分布->6个期望，一个3*3的协方差矩阵
%% 算均值
mu=zeros(3,2);
mu(:,1)=mean(white_set);
mu(:,2)=mean(red_set);
%% 算协方差矩阵
cov_white=cov(white_set);
cov_red=cov(red_set);
subplot(2,3,2);
imagesc(blkdiag(corrcoef(white_set),corrcoef(red_set)));    
colorbar;
title('RGB的相关系数矩阵');
%% 计算先验概率和新样本的条件概率
%各向量第一列(1)为白色对应参数
prior_P=[sum(train_set(:,5)==-1)/size(train_set(:,5),1),sum(train_set(:,5)==1)/size(train_set(:,5),1)];
%
mask_rgb(:,:,1)=mask;
mask_rgb(:,:,2)=mask;
mask_rgb(:,:,3)=mask;
nemo_rgb=double(reshape(img_rgb_masked(mask_rgb>0),[7364,3]))/255;
% LP(:,1)=mvnpdf(double(nemo_rgb),mu(:,1)',cov_white);
% LP(:,2)=mvnpdf(double(nemo_rgb),mu(:,2)',cov_red);
%% 最大化后验概率
post_P=zeros(size(nemo_rgb,1),2);
% post_P(:,1)=prior_P(1)*LP(:,1);
% post_P(:,2)=prior_P(1)*LP(:,2);
for i=1:size(nemo_rgb,1)
    post_P(i,1)=(-1/2)*(nemo_rgb(i,:)-mu(:,1)')/cov_white*((nemo_rgb(i,:)-mu(:,1)'))'-(1/2)*log(det(cov_white))+log(prior_P(1));    
    post_P(i,2)=(-1/2)*(nemo_rgb(i,:)-mu(:,2)')/(cov_red)*((nemo_rgb(i,:)-mu(:,2)'))'-(1/2)*log(det(cov_red))+log(prior_P(2));
end
pixel_class=zeros(size(nemo_rgb,1),1);
pixel_class(post_P(:,1)>=post_P(:,2))=-1;
pixel_class(post_P(:,1)<post_P(:,2))=1;
%% 显示
[index(:,1),index(:,2),~]=find(mask==1);
class_mask=mask;
for i=1:1:size(index,1)
    class_mask(index(i,1),index(i,2))=pixel_class(i);
end
nemo_binary1=double(img(:,:,1)).*mask;
nemo_binary2=double(img(:,:,2)).*mask;
nemo_binary3=double(img(:,:,3)).*mask;
%强调红
nemo_binary1(class_mask==1)=255;
% nemo_binary2(class_mask==1)=0;
% nemo_binary3(class_mask==1)=0;
%全白
nemo_binary1(class_mask==-1)=255;
nemo_binary2(class_mask==-1)=255;
nemo_binary3(class_mask==-1)=255;
%
nemo_binary(:,:,1)=nemo_binary1;
nemo_binary(:,:,2)=nemo_binary2;
nemo_binary(:,:,3)=nemo_binary3;
%
nemo_binary=uint8(nemo_binary);
subplot(2,3,4)
imshow(nemo_binary);
title('分割后的nemo');
subplot(2,3,6);
scatter3(nemo_rgb(pixel_class==-1,1),nemo_rgb(pixel_class==-1,2),nemo_rgb(pixel_class==-1,3),'.b');
hold on
scatter3(nemo_rgb(pixel_class==1,1),nemo_rgb(pixel_class==1,2),nemo_rgb(pixel_class==1,3),'.r');
scatter3(mu(1,1),mu(2,1),mu(3,1),200,'.b'); 
text(mu(1,1),mu(2,1),mu(3,1),'白色分布中心');
hold on
scatter3(mu(1,2),mu(2,2),mu(3,2),200,'.r'); 
text(mu(1,2),mu(2,2),mu(3,2),'红色分布中心');
hold off
title('分类器结果');


impixelinfo







%% 性能评估
%拿训练集数据及标签进行评估
LP_t(:,1)=mvnpdf(double(train_set(:,2:4)),mu(:,1)',cov_white);
LP_t(:,2)=mvnpdf(double(train_set(:,2:4)),mu(:,2)',cov_red);
post_P_t(:,1)=prior_P(1)*LP_t(:,1);
post_P_t(:,2)=prior_P(1)*LP_t(:,2);
pixel_class_t=zeros(size(train_set(:,2:3),1),1);
pixel_class_t(post_P_t(:,1)>=post_P_t(:,2))=-1;
pixel_class_t(post_P_t(:,1)<post_P_t(:,2))=1;
pixel_class_t(pixel_class_t==-1)=0;
train_set(train_set(:,5)==-1,5)=0;
subplot(2,3,2)
text(0,8,['准确率为 ',num2str(1-(sum(pixel_class_t~=train_set(:,5)))/length(pixel_class_t),2)]); 
%
pixel_score_t=logsig(post_P_t(:,2)-post_P_t(:,1));
pixel_class_t(pixel_score_t>0.5)=1;
pixel_class_t(pixel_score_t<=0.5)=0;
train_set(train_set(:,5)==-1,5)=0;
%画ROC
[X,Y,T,AUC] =perfcurve(train_set(:,5),pixel_score_t,'1');
figure('NumberTitle', 'off', 'Name', 'ROC & PR'); 
subplot(1,2,1);
plot(X,Y)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification by Bayesian Decision (RGB)')
%由于像素值的分布符合多维正态分布，所以贝叶斯决策和EM效果很好
text(0.6,0.4,['AUC= ',num2str(AUC)]);

[~,best_threshold_index]=max(1-X+Y);
best_threshhold=T(best_threshold_index);
%画PR曲线
[X,Y] =perfcurve(train_set(:,5),pixel_score_t,'1','xCrit', 'reca', 'yCrit', 'prec');
subplot(1,2,2);
plot(X,Y)
xlabel('Recall') 
ylabel('Precision')
title('PRC for Classification by Bayesian Decision (RGB)')
%正确率
acc=sum(pixel_class_t==train_set(:,5))/size(train_set,1)
%%
%计算最佳阈值处分类器正确率，TP+(1-FP)和最大，但acc不一定最大
pixel_class_t(pixel_score_t>best_threshhold)=1;
pixel_class_t(pixel_score_t<=best_threshhold)=0;
acc=sum(pixel_class_t==train_set(:,5))/size(train_set,1)


%% 问题
%由于红色类别的b分量一直很小，所以一旦b分量增大，就很可能跨越二次曲面决策面变为白色类造成误判
%(-1/2)()


t1=cputime-t0;