%E:\OnlineDisk\OneDrive\OneDrive - Neuroinformatics Collaboratory\Github\Pattern Recognition\SVM
%% 目标函数/成本函数
function Y=fun_cost(X,scale,train_data,train_label)
%     [m,~]=size(X);
%     Y=ones(m,1);
%     %Y=(X(:,1)-1).^2+(X(:,4)-4).^2+(X(:,2)-2).^2+(X(:,3)-3).^2;
%     Y=0.5*((X(:,1).*X(:,4))-(X(:,2).*X(:,3)));
%     Y=Y+fun_penalty(X,scale);
    [m,~]=size(X);
    Y=ones(m,1);
    for i=1:m
        cmd= ['-t 2 -v ',num2str(5),' -c ',num2str(X(i,1)),' -g ',num2str(X(i,2))];
        Y(i)=-svmtrain(train_label,train_data,cmd);     
        %Y=Y+fun_penalty(X,scale);
    end
   
end