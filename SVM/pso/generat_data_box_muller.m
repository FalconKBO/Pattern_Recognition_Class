%E:\OnlineDisk\OneDrive\OneDrive - Neuroinformatics Collaboratory\Github\Pattern Recognition\SVM
clear;clc;close all;
u1=rand(1,1000);
u2=rand(1,1000);
r1=sqrt(-2*log(u1))+3;
r2=sqrt(-2*log(u2))+6;
theta1=2*pi*u2;
theta2=2*pi*u1;
x1=r1.*cos(theta1);
y1=r1.*sin(theta1);
x2=r2.*cos(theta2);
y2=r2.*sin(theta2);
% plot(x1,y1,'.')
% hold on
% plot(x2,y2,'.')
train_data=[[x1',y1'];[x2',y2']];
plot(train_data(:,1),train_data(:,2),'.')
%save .\own_data\train_data.mat train_data
train_label=[-ones(1000,1);ones(1000,1)];
%save .\own_data\train_label.mat train_label
clear;figure;
u1=rand(1,500);
u2=rand(1,500);
r1=sqrt(-2*log(u1))+3;
r2=sqrt(-2*log(u2))+6;
theta1=2*pi*u2;
theta2=2*pi*u1;
x1=r1.*cos(theta1);
y1=r1.*sin(theta1);
x2=r2.*cos(theta2);
y2=r2.*sin(theta2);
% plot(x1,y1,'.')
% hold on
% plot(x2,y2,'.')
test_data=[[x1',y1'];[x2',y2']];
plot(test_data(:,1),test_data(:,2),'.')
%save .\own_data\test_data.mat test_data
test_label=[-ones(500,1);ones(500,1)];
%save .\own_data\test_label.mat test_label
