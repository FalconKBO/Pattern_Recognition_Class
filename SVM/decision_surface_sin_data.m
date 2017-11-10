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
x=0:4*pi/1999:4*pi;
y1=0.5+0.6*sin(x);
y2=0.6*sin(x)-0.5;
train_data=[x;y1]';
train_data=[train_data;[x;y2]'];
train_label=[ones(2000,1);-ones(2000,1)];
x=0:4*pi/499:4*pi;
y1=0.5+0.6*sin(x);
y2=0.6*sin(x)-0.5;
test_data=[x;y1]';
test_data=[test_data;[x;y2]'];
test_label=[ones(500,1);-ones(500,1)];
figure('NumberTitle', 'off', 'Name', 'svm for sin data');
%% Ԥ����;
%��һ��
% train_data=train_data./(max(max(abs(train_data)),max(abs(test_data)))-min(min(abs(train_data)),min(abs(test_data))));
% test_data=test_data./(max(max(abs(train_data)),max(abs(test_data)))-min(min(abs(train_data)),min(abs(test_data))));
%% train & predict
%pso�Ż���Ĳ��� [38644.5518185840,0.221398585898100]
c_best=10.5518185840;
gamma_best=0.221398585898100;

% for gn_num=1:1:40
%     gn_num
%     c_best=0.5^(21-gn_num);
%     %gamma_best=0.5^(21-gn_num);


cmd= ['-t 2 -c ',num2str(c_best),' -g ',num2str(gamma_best)];
%ѵ��
model=svmtrain(train_label,train_data,cmd);
save .\own_data\fun_svm_predict.mat model
%ѵ�������
cmd=strcat('-v 5 ',cmd);
acc=svmtrain(train_label,train_data,cmd);
%����
[predicted_label, accuracy, decision_values] = svmpredict(test_label,test_data,model);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% һ����Χ�����ݵķ���������ֲ�
x=[-4*pi:12*pi/99:8*pi]';
y=[-6:12/99:6]';
[X,Y]=meshgrid(x,y);
X=reshape(X,[size(x).^2,1]);
Y=reshape(Y,[size(x).^2,1]);
Z=fun_svm_predict([X,Y]);
X1=reshape(X,[size(x,1),size(x,1)]);
Y1=reshape(Y,[size(y,1),size(y,1)]);
Z1=reshape(Z,[size(x,1),size(y,1)]);

subplot(1,3,1);
surfc(X1,Y1,Z1,'EdgeColor','none');
hold on
contour3(X1,Y1,Z1,'Color','g');
xlabel('X');ylabel('Y');zlabel('Z');title('��SVM���');
hold off
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% draw result 
inlayer=test_data(test_label==1,1:end);
label_true{1}=test_label(test_label==1);
outlayer=test_data(test_label==-1,1:end);
label_true{2}=test_label(test_label==-1);
subplot(1,3,2);
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
xlabel('X');ylabel('Y');title('�����Էֽ����֧�����������ı߽�');
hold off
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%�⺯����㣬�ظ�����ܶ�
% pred_data=test_data;
% pred_result=fun_svm_predict(sv(5,:));
% %ͨ���洦���ݴ���
% 
% %�����
% %�������÷��̵���㣬��֧���������������
% %�����������ݵı���������ԣ�����֧�������İ뾶�����ڸ�Բ�ܴ�������㣬ͬ������Ҳ���Ի���������֧�������߽磨�����ݴ�?����
% %x_decision0=[-pi:6*pi/199:5*pi;0.25*sin([-1:5*pi/199:5*pi])]';
% %x_decision0=[-pi:6*pi/199:5*pi;zeros(1,200)]';
% %x_decision0=[-20*pi:40*pi/199:20*pi;0.25*sin([-20*pi:40*pi/199:20*pi])]';
% x_decision0=[-4*pi:12*pi/199:8*pi;zeros(1,200)]';
% % x_decision0=x_decision0./(max(max(abs(x_decision0)),max(abs(x_decision0)))-min(min(abs(x_decision0)),min(abs(x_decision0))));
% %�ֽ���(���ͳ�ֵ���ú��й�ϵ)
% x_decision=fsolve(@fun_svm_predict,x_decision0,optimset('Display','off','Algorithm','Levenberg-Marquardt','TolFun',1e-2));
% % for i=1:1:size(x_decision0)
% % %x_decision(i,:)=fzero('fun_svm_predict',x_decision0(i,:));
% % %x_decision(i,:)=fmincon(@fun_svm_predict,x_decision0(i,:),[],[],[],[],[-1,-2],[1,2],[],optimset('Display','off','Algorithm','sqp'));
% % end
% x_decision=sortrows(x_decision,1);
% plot(x_decision(:,1),x_decision(:,2),'-x');
% hold on
% %��edge1
% fun_svm_edge1=@(data)(fun_svm_predict(data)-1);
% x_edge1=fsolve(fun_svm_edge1,x_decision,optimset('Display','off','Algorithm','Levenberg-Marquardt','TolFun',1e-2));
% x_edge1=sortrows(x_edge1,1);
% plot(x_edge1(:,1),x_edge1(:,2),'-x');
% hold on
% %��edge2
% fun_svm_edge2=@(data)(fun_svm_predict(data)+1);
% x_edge2=fsolve(fun_svm_edge2,x_decision,optimset('Display','off','Algorithm','Levenberg-Marquardt','TolFun',1e-2));
% x_edge2=sortrows(x_edge2,1);
% plot(x_edge2(:,1),x_edge2(:,2),'-x');
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% draw ROC
% figure('NumberTitle', 'off', 'Name', 'ROC');
subplot(1,3,3);
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






%    %% ��ͼ
%    picname=[num2str(gn_num) '.fig'];%������ļ�������i=1ʱ��picname=1.fig
%    saveas(gcf,picname)
%    pause(0.1)
% 
% end





rmpath .\pso


