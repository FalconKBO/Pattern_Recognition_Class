%E:\OnlineDisk\OneDrive\OneDrive - Neuroinformatics Collaboratory\Github\Pattern Recognition\SVM
function theta=fun_theta(data)
%% ��������theta���������ݵ�֮��ľ���
%% Ҳ������ͨ�����ؾ�����Ϊ����֮�����Ķ�����Ȼ�����LLE����չ�����򣬷��㻭�߽�
x(:,1)=data(:,1);
y(:,1)=data(:,2);
theta=atan(y./x);
theta(x<0)=pi+theta(x<0);

end