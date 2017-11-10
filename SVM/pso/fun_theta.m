%E:\OnlineDisk\OneDrive\OneDrive - Neuroinformatics Collaboratory\Github\Pattern Recognition\SVM
function theta=fun_theta(data)
%% 这里用了theta来衡量数据点之间的距离
%% 也可以用通过算测地距来作为数据之间距离的度量，然后类比LLE进行展开排序，方便画边界
x(:,1)=data(:,1);
y(:,1)=data(:,2);
theta=atan(y./x);
theta(x<0)=pi+theta(x<0);

end