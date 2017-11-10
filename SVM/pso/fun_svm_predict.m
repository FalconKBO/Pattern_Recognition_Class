%E:\OnlineDisk\OneDrive\OneDrive - Neuroinformatics Collaboratory\Github\Pattern Recognition\SVM
function zero_data=fun_svm_predict(data)
    load .\own_data\fun_svm_predict.mat
    
    sv=full(model.SVs);
    sv_coef=model.sv_coef;
    rho=model.rho;
    gamma=model.Parameters(4);


    for i=1:1:size(data,1)
        x_diff=(repmat(data(i,:),[size(sv,1),1])-sv);
        x_norm=sum((x_diff.^2)')';
        %kernel_x=exp(-x_norm/(gamma_best^2));
        kernel_x=exp(-x_norm*gamma);
        zero_data(i,:)=sum(kernel_x.*sv_coef)-rho;
    end
end