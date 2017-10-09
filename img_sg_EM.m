%E:\OnlineDisk\OneDrive\OneDrive - Neuroinformatics Collaboratory\Github\Pattern Recognition\image segmentation
%% ��ȡ����
%% ���ݷֲ�����->��˹�ֲ� �������->����
%% �趨��ʼ����->�������
%% ѭ������E����M����ֱ����������
 %% E������С���ձ�Ҷ˹���ߣ������ü���Ĳ����͵�������Ĳ������з���
 %% M���������Ȼ���Ʋ�����������M���Ľ�����������µĲ���
 %% �޷��������ݶȲ���򵽴��趨�������˳�
%% ��������


%% ��ȡ����
clear;clc;close all;
t0=cputime;
%·��
file_path=mfilename('fullpath');
i=strfind(file_path,'\');
file_path=file_path(1:i(end));
cd(file_path);
%��ͼ����ɰ�
img=imread('.\data\309.bmp');
mask_file=load('.\data\Mask.mat','-mat','Mask');
mask=mask_file.Mask;
nemo_size=sum(sum(mask==1));
img_rgb_masked=img;
img_rgb_masked(:,:,1)=double(img(:,:,1)).*mask;
img_rgb_masked(:,:,2)=double(img(:,:,2)).*mask;
img_rgb_masked(:,:,3)=double(img(:,:,3)).*mask;
figure('NumberTitle', 'off', 'Name', 'Expectation Maximization Algorithm'); 
subplot(2,3,1);
imshow(img_rgb_masked);
title('��ɫ��nemo');
   
%% �ҵ�����������
mask_rgb(:,:,1)=mask;
mask_rgb(:,:,2)=mask;
mask_rgb(:,:,3)=mask;
nemo_rgb=double(reshape(img_rgb_masked(mask_rgb>0),[7364,3]))/255;
% subplot(2,3,3);
% scatter3(nemo_rgb(:,1),nemo_rgb(:,2),nemo_rgb(:,3),'.k');
% title('��������������')
%% ���ݷֲ�����->��˹�ֲ� �������->����
%% �趨��ʼ����->�������
prior_P=[0.2,0.8];%% �����ֵ��ȷ��
%Ϊ���������������Ը��ݱ�׼��ɫ�Ͱ�ɫ��RGB������������ľ�ֵ
%mu=[255,255,255;255,0,0]'/255;
%mu=[15,10,10;0,0,0]'/255;
mu=[100,100,100;5,5,5]'/255;
%mu=rand(3,2)*200;%���ܲ�����

% cov_white=pascal(3)/6;
% cov_red=pascal(3)/6;
cov_white=eye(3)/2;
cov_red=eye(3)/2;
% cov_white=[0.0169632553112706,0.0183352856625498,0.0218360851344370;0.0183352856625498,0.0273182191857538,0.0387163373779709;0.0218360851344370,0.0387163373779709,0.0644438888731153];
% cov_red=[0.0292777730104772,0.0125593765248477,0.00271680696273606;0.0125593765248477,0.00902524900213047,0.00659359911802614;0.00271680696273606,0.00659359911802614,0.0111114220397838];


%% ѭ������E����M����ֱ����������-> 
%��ʼ��
gn_num=1;
gn_num_max=100;
t=1:1:gn_num_max;
grdt=zeros(1,gn_num_max);
grdt_min=10^(-3);
LP=zeros(size(nemo_rgb,1),2);
LP_SUM=zeros(gn_num_max,3);
u_times=zeros(gn_num_max,6);
LP_SUM_DIFF=ones(gn_num_max,2)*10^(6);
post_P=zeros(size(nemo_rgb,1),2);
pixel_class=zeros(size(nemo_rgb,1),1);
%����
while gn_num<gn_num_max 
%% E��    
    LP(:,1)=log(mvnpdf(nemo_rgb,mu(:,1)',cov_white));
    LP(:,2)=log(mvnpdf(nemo_rgb,mu(:,2)',cov_red)); 
    
    %��¼��Ȼ����ֵ�仯������ֵ�仯
    LP_SUM(gn_num,1)=sum(LP(LP(:,1)>=LP(:,2),1));
    LP_SUM(gn_num,2)=sum(LP(LP(:,1)<LP(:,2),2));
    LP_SUM(gn_num,3)=LP_SUM(gn_num,1)+LP_SUM(gn_num,2);
    u_times(gn_num,1:3)=mu(:,1);
    u_times(gn_num,4:6)=mu(:,2);    
    if gn_num>1
        %����Ȼ���ʱ仯
        LP_SUM_DIFF(gn_num,:)=LP_SUM(gn_num,1:2)-LP_SUM(gn_num-1,1:2);
        %�������仯����
        subplot(2,3,5);  
        plot(t(1:gn_num),u_times(1:gn_num,1:3)*255,'b'); 
        hold on
        plot(t(1:gn_num),u_times(1:gn_num,4:6)*255,'r'); 
        hold off
        title('��ֵ�仯');
        %��������Ȼ���ʱ仯����
        subplot(2,3,3); 
        plot(t(1:gn_num),LP_SUM(1:gn_num,1)+LP_SUM(1:gn_num,2),'g');
        lp_curve_color='brg';
        for i=1:1:length(lp_curve_color)
            lp_curve(i)=plot(t(1:gn_num),LP_SUM(1:gn_num,i),lp_curve_color(i) );            
            hold on
        end
        legend(lp_curve,'���ף���Ȼ����','���죩��Ȼ����','��Ȼ���ʺ�', 'Location','southeast' );
        hold off
        title('������Ȼ����') 
    end  
    %��������
    post_P(:,1)=LP(:,1)+log(prior_P(1));
    post_P(:,2)=LP(:,2)+log(prior_P(2)); 
    %����     
    pixel_class(post_P(:,1)>=post_P(:,2))=-1;
    pixel_class(post_P(:,1)<post_P(:,2))=1;
    %% M��
    white_set=nemo_rgb([(pixel_class==-1),(pixel_class==-1),(pixel_class==-1)]);
    white_set=reshape(white_set,length(white_set)/3,3);
    red_set=nemo_rgb([(pixel_class==1),(pixel_class==1),(pixel_class==1)]);
    red_set=reshape(red_set,length(red_set)/3,3);
    mu(:,1)=mean(white_set);
    mu(:,2)=mean(red_set);
    cov_white=cov(white_set);
    cov_red=cov(red_set);
    prop=[sum(pixel_class==-1)/length(pixel_class),sum(pixel_class==1)/length(pixel_class)];
    subplot(2,3,2);
    imagesc(blkdiag(corrcoef(white_set),corrcoef(red_set)));    
    colorbar;
    title('RGB�����ϵ������');
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
    scatter3(mu(1,1),mu(2,1),mu(3,1),200,'.g'); 
    text(mu(1,1),mu(2,1),mu(3,1),'��ɫ�ֲ�����');
    hold on
    scatter3(mu(1,2),mu(2,2),mu(3,2),200,'.b'); 
    text(mu(1,2),mu(2,2),mu(3,2),'��ɫ�ֲ�����');
    hold on
    scatter3(nemo_rgb(pixel_class==-1,1),nemo_rgb(pixel_class==-1,2),nemo_rgb(pixel_class==-1,3),20,'.b');
    hold on
    scatter3(nemo_rgb(pixel_class==1,1),nemo_rgb(pixel_class==1,2),nemo_rgb(pixel_class==1,3),20,'.r');
    title('������'); 
    hold off;
    subplot(2,3,4);
    text(-170,350,['�� ',int2str(gn_num),' ��' ,'  �����Ϊ ',num2str(prop(1),2),'��',num2str(prop(2),2)]);  
    hold off;    
  %% �ж�
    %��e�˳�
    if strcmpi(get(gcf,'CurrentCharacter'),'e')
        break;
    end       
    %�ݶȲ������˳�
    if (abs(LP_SUM_DIFF(gn_num,1))<grdt_min && abs(LP_SUM_DIFF(gn_num,2))<grdt_min)
       disp(['��',int2str(gn_num),'���ݶȱ仯С����ֵ����������']);
       break;       
    end
    %% ��ͼ
%      picname=[num2str(gn_num) '.fig'];%������ļ�������i=1ʱ��picname=1.fig
%      saveas(gcf,picname)
    %% 
     pause(0.05)
     gn_num=gn_num+1;
     hold off all;
   
  
end %ѵ������
impixelinfo; 






%% ��������
%��ѵ�������ݼ���ǩ��������
figure('NumberTitle', 'off', 'Name', 'ROC & PR'); 
train_set_file=load('.\data\array_sample.mat');
train_set=train_set_file.array_sample;
LP_t(:,1)=mvnpdf(double(train_set(:,2:4)),mu(:,1)',cov_white);
LP_t(:,2)=mvnpdf(double(train_set(:,2:4)),mu(:,2)',cov_red);
post_P_t(:,1)=prior_P(1)*LP_t(:,1);
post_P_t(:,2)=prior_P(2)*LP_t(:,2);
pixel_class_t=zeros(size(train_set(:,2:3),1),1);
pixel_score_t=logsig(post_P_t(:,2)-post_P_t(:,1));
pixel_class_t(pixel_score_t>0.5)=1;
pixel_class_t(pixel_score_t<=0.5)=0;
train_set(train_set(:,5)==-1,5)=0;
%��ROC
[X,Y,T,AUC] =perfcurve(train_set(:,5),pixel_score_t,'1');

subplot(1,2,1);
plot(X,Y)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification by Expectation Maximization Cluster')
%��������ֵ�ķֲ����϶�ά��̬�ֲ������Ա�Ҷ˹���ߺ�EMЧ���ܺ�
text(0.6,0.4,['AUC= ',num2str(AUC)]);
%��¼�����ֵ������EM�൱���ǰ�����ֵ=0.5�����ģ����������ֵ��0.5������С�ķ�Χ��
[~,best_threshhold_index]=max(1-X+Y);
best_threshold=T(best_threshhold_index);
%��PR����
[X,Y] =perfcurve(train_set(:,5),pixel_score_t,'1','xCrit', 'reca', 'yCrit', 'prec');
subplot(1,2,2);
plot(X,Y)
xlabel('Recall') 
ylabel('Precision')
title('PRC for Classification by Expectation Maximization Cluster')
%��ȷ��
acc=sum(pixel_class_t==train_set(:,5))/size(train_set,1)
%%

% %���������ֵ����������ȷ�ʣ�TP+(1-FP)����󣬵�acc��һ�����
% pixel_class_t(pixel_score_t>best_threshold)=1;
% pixel_class_t(pixel_score_t<=best_threshold)=0;
% acc=sum(pixel_class_t==train_set(:,5))/size(train_set,1)

% % ���������ֵ�����Ȼ���ʲ�ֵ����������Ȼ����ֵ����µ�������ʣ��ͷ��������ı�����һ������
% prior_P_t(2)=(log(best_threshhold/(1-best_threshhold))+LP_t(6,2))/LP_t(6,1)+LP_t(6,2);%(ֻȡ��һ��)
% prior_P_t(1)=1-prior_P_t(2);
% post_P_t(:,1)=prior_P_t(1)*LP_t(:,1);
% post_P_t(:,2)=prior_P_t(2)*LP_t(:,2);
% pixel_class_t=zeros(size(train_set(:,2:3),1),1);
% pixel_score_t=logsig(post_P_t(:,2)-post_P_t(:,1));
% pixel_class_t(pixel_score_t>0.5)=1;
% pixel_class_t(pixel_score_t<=0.5)=0;
% acc=sum(pixel_class_t==train_set(:,5))/size(train_set,1)

t1=cputime-t0;