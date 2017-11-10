%E:\OnlineDisk\OneDrive\OneDrive - Neuroinformatics Collaboratory\Github\Pattern Recognition\image segmentation
%% ��ȡͼƬ��mask
%% ��ȡѵ�����������ֲ�
%% ����ѵ������������ֲ��Ĳ���
%% ����ѵ���������������
%% ��������������������
%% ��󻯺�����ʷ���
%% ��ʾ

%% ��������
%%��ȡ����
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
figure('NumberTitle', 'off', 'Name', '��rgb�ָ�nemo'); 
subplot(2,3,1);
imshow(img_rgb_masked);
title('ԭ����nemo');
%% ��ȡ������ѵ�����ֲ�
train_set_file=load('.\data\array_sample.mat');
train_set=train_set_file.array_sample;
white_set=train_set(train_set(:,5)==-1,2:4);
red_set=train_set(train_set(:,5)==1,2:4);
%% RGB���߹�������ά����̬�ֲ�,��������άRGB�ռ��ɢ��ͼ
%scatter3(train_set(:,1),train_set(:,2),train_set(:,3),'.');
subplot(2,3,3);
scatter3(white_set(:,1),white_set(:,2),white_set(:,3),'.b');
hold on
scatter3(red_set(:,1),red_set(:,2),red_set(:,3),'.r');
title('ѵ��������')
%% ��������ֲ��Ĳ�����������ά��˹�ֲ�->6��������һ��3*3��Э�������
%% ���ֵ
mu=zeros(3,2);
mu(:,1)=mean(white_set);
mu(:,2)=mean(red_set);
%% ��Э�������
cov_white=cov(white_set);
cov_red=cov(red_set);
subplot(2,3,2);
imagesc(blkdiag(corrcoef(white_set),corrcoef(red_set)));    
colorbar;
title('RGB�����ϵ������');
%% ����������ʺ�����������������
%��������һ��(1)Ϊ��ɫ��Ӧ����
prior_P=[sum(train_set(:,5)==-1)/size(train_set(:,5),1),sum(train_set(:,5)==1)/size(train_set(:,5),1)];
%
mask_rgb(:,:,1)=mask;
mask_rgb(:,:,2)=mask;
mask_rgb(:,:,3)=mask;
nemo_rgb=double(reshape(img_rgb_masked(mask_rgb>0),[7364,3]))/255;
% LP(:,1)=mvnpdf(double(nemo_rgb),mu(:,1)',cov_white);
% LP(:,2)=mvnpdf(double(nemo_rgb),mu(:,2)',cov_red);
%% ��󻯺������
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
%% ��ʾ
[index(:,1),index(:,2),~]=find(mask==1);
class_mask=mask;
for i=1:1:size(index,1)
    class_mask(index(i,1),index(i,2))=pixel_class(i);
end
nemo_binary1=double(img(:,:,1)).*mask;
nemo_binary2=double(img(:,:,2)).*mask;
nemo_binary3=double(img(:,:,3)).*mask;
%ǿ����
nemo_binary1(class_mask==1)=255;
% nemo_binary2(class_mask==1)=0;
% nemo_binary3(class_mask==1)=0;
%ȫ��
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
title('�ָ���nemo');
subplot(2,3,6);
scatter3(nemo_rgb(pixel_class==-1,1),nemo_rgb(pixel_class==-1,2),nemo_rgb(pixel_class==-1,3),'.b');
hold on
scatter3(nemo_rgb(pixel_class==1,1),nemo_rgb(pixel_class==1,2),nemo_rgb(pixel_class==1,3),'.r');
scatter3(mu(1,1),mu(2,1),mu(3,1),200,'.b'); 
text(mu(1,1),mu(2,1),mu(3,1),'��ɫ�ֲ�����');
hold on
scatter3(mu(1,2),mu(2,2),mu(3,2),200,'.r'); 
text(mu(1,2),mu(2,2),mu(3,2),'��ɫ�ֲ�����');
hold off
title('���������');


impixelinfo







%% ��������
%��ѵ�������ݼ���ǩ��������
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
text(0,8,['׼ȷ��Ϊ ',num2str(1-(sum(pixel_class_t~=train_set(:,5)))/length(pixel_class_t),2)]); 
%
pixel_score_t=logsig(post_P_t(:,2)-post_P_t(:,1));
pixel_class_t(pixel_score_t>0.5)=1;
pixel_class_t(pixel_score_t<=0.5)=0;
train_set(train_set(:,5)==-1,5)=0;
%��ROC
[X,Y,T,AUC] =perfcurve(train_set(:,5),pixel_score_t,'1');
figure('NumberTitle', 'off', 'Name', 'ROC & PR'); 
subplot(1,2,1);
plot(X,Y)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification by Bayesian Decision (RGB)')
%��������ֵ�ķֲ����϶�ά��̬�ֲ������Ա�Ҷ˹���ߺ�EMЧ���ܺ�
text(0.6,0.4,['AUC= ',num2str(AUC)]);

[~,best_threshold_index]=max(1-X+Y);
best_threshhold=T(best_threshold_index);
%��PR����
[X,Y] =perfcurve(train_set(:,5),pixel_score_t,'1','xCrit', 'reca', 'yCrit', 'prec');
subplot(1,2,2);
plot(X,Y)
xlabel('Recall') 
ylabel('Precision')
title('PRC for Classification by Bayesian Decision (RGB)')
%��ȷ��
acc=sum(pixel_class_t==train_set(:,5))/size(train_set,1)
%%
%���������ֵ����������ȷ�ʣ�TP+(1-FP)����󣬵�acc��һ�����
pixel_class_t(pixel_score_t>best_threshhold)=1;
pixel_class_t(pixel_score_t<=best_threshhold)=0;
acc=sum(pixel_class_t==train_set(:,5))/size(train_set,1)


%% ����
%���ں�ɫ����b����һֱ��С������һ��b�������󣬾ͺܿ��ܿ�Խ��������������Ϊ��ɫ���������
%(-1/2)()


t1=cputime-t0;