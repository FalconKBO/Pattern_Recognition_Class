%E:\OnlineDisk\OneDrive\OneDrive - Neuroinformatics Collaboratory\Github\Pattern Recognition\image segmentation
%% ��ȡͼƬ��mask
%% ��ȡѵ�����������ֲ�
%% ����ѵ������������ֲ��Ĳ���
%% ����ѵ���������������
%% ��������������������
%% ��󻯺�����ʷ���
%% ��ʾ
%% ��������

%% ��ȡͼƬ��mask
clear;clc;close all;
t0=cputime;
file_path=mfilename('fullpath');
i=strfind(file_path,'\');
file_path=file_path(1:i(end));
cd(file_path);
img=imread('.\data\309.bmp');
mask_file=load('.\data\Mask.mat','-mat','Mask');
mask=mask_file.Mask;
img_gray=rgb2gray(img);
img_gray_mask=img_gray.*uint8(mask);
figure('NumberTitle', 'off', 'Name', '���Ҷȷָ�nemo');  
subplot(2,2,1);
imshow(img_gray_mask);
title('�ҶȻ���nemo');
%% ��ȡѵ�����������ֲ�
train_set_file=load('.\data\array_sample.mat');
train_set=train_set_file.array_sample;
white_set=train_set(train_set(:,5)==-1,1);
red_set=train_set(train_set(:,5)==1,1);
[n_pdf,n_value]=ksdensity(train_set(:,1));
subplot(2,2,2);
plot(n_value*255,n_pdf);
title('����ֵ�ֲ�');
hold on
%% ��������ֲ��Ĳ���
mu=zeros(1,2);
sigma=ones(1,2);
x = 0:0.01:1.2;
%% white label==-1
mu(1)=mean(white_set);
sigma(1)=std(white_set);
%% red  label==1
mu(2)=mean(red_set);
sigma(2)=std(red_set);
%% ����������ʺ�����������������
prior_P=[sum(train_set(:,5)==-1)/size(train_set(:,5),1),sum(train_set(:,5)==1)/size(train_set(:,5),1)];
%��������
nemo_gray=img_gray_mask(mask>0);%����
[index(:,1),index(:,2),~]=find(mask==1);
nemo_gray=double(nemo_gray)/255;
LP(:,1)=normpdf(double(nemo_gray),mu(1),sigma(1));
LP(:,2)=normpdf(double(nemo_gray),mu(2),sigma(2));
%��ͼ
y = prior_P(1)*normpdf(x, mu(1),sigma(1));
plot(x*255,y);
hold on
y = prior_P(2)*normpdf(x, mu(2),sigma(2));
plot(x*255,y);
%% ��󻯺�����ʷ���
post_P=[LP(:,1)*prior_P(1),LP(:,2)*prior_P(2)];
pixel_class=zeros(length(nemo_gray),1);
pixel_class(post_P(:,1)>=post_P(:,2))=-1;
pixel_class(post_P(:,1)<post_P(:,2))=1;
%% ��ʾ
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
subplot(2,2,3)
imshow(nemo_binary);
title('�ָ���nemo')
impixelinfo


%% ��������
%��ѵ�������ݼ���ǩ��������
LP_t(:,1)=normpdf(double(train_set(:,1)),mu(1)');
LP_t(:,2)=normpdf(double(train_set(:,1)),mu(2)');
post_P_t(:,1)=prior_P(1)*LP_t(:,1);
post_P_t(:,2)=prior_P(1)*LP_t(:,2);
pixel_class_t=zeros(size(train_set(:,2:3),1),1);
pixel_class_t(post_P_t(:,1)>=post_P_t(:,2))=-1;
pixel_class_t(post_P_t(:,1)<post_P_t(:,2))=1;
pixel_class_t(pixel_class_t==-1)=0;
train_set(train_set(:,5)==-1,5)=0;
subplot(2,2,2)
text(-0.5,-0.5,['׼ȷ��Ϊ ',num2str(1-(sum(pixel_class_t~=train_set(:,5)))/length(pixel_class_t),2)]); 
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
title('ROC for Classification by Bayesian Decision (Gray)')
%��������ֵ�ķֲ����϶�ά��̬�ֲ������Ա�Ҷ˹���ߺ�EMЧ���ܺ�
text(0.6,0.4,['AUC= ',num2str(AUC)]);

[~,best_threshhold_index]=max(1-X+Y);
best_threshhold=T(best_threshhold_index);
%��PR����
[X,Y] =perfcurve(train_set(:,5),pixel_score_t,'1','xCrit', 'reca', 'yCrit', 'prec');
subplot(1,2,2);
plot(X,Y)
xlabel('Recall') 
ylabel('Precision')
title('PRC for Classification by  Bayesian Decision (Gray)')
%��ȷ��
acc=sum(pixel_class_t==train_set(:,5))/size(train_set,1)

%%
%���������ֵ����������ȷ�ʣ�TP+(1-FP)����󣬵�acc��һ�����
pixel_class_t(pixel_score_t>best_threshhold)=1;
pixel_class_t(pixel_score_t<=best_threshhold)=0;
acc=sum(pixel_class_t==train_set(:,5))/size(train_set,1)




t1=cputime-t0;

%% ����
%��ͨ���ķ�����Ϣ�����٣����ڷ����ԭ�򣬿������ĳЩ�߻Ҷ��������гɰ�ɫ
