%E:\OnlineDisk\OneDrive\OneDrive - Neuroinformatics Collaboratory\Github\Pattern Recognition\SVM
stepall=40;%ͼ����֡��
for i=1:stepall
    picname=[num2str(i) '.fig'];
    open(picname)
     set(gcf,'outerposition',get(0,'screensize'));% matlab�������
    frame=getframe(gcf);  
    im=frame2im(frame);%����gif�ļ���ͼ�������index����ͼ��  
    %[I,map]=rgb2ind(im,20);  
    [I,map]=rgb2ind(im,256,'nodither');  %���Բ鿴rbg2indѡ����Ҫ��ת����ʽ������λ��
    if i==1
        imwrite(I,map,'SVM_RBF_xor.gif','gif', 'Loopcount',inf,'DelayTime',0.2);%��һ�α��봴����
    elseif i==stepall
        imwrite(I,map,'SVM_RBF_xor.gif','gif','WriteMode','append','DelayTime',0.2);
    else
        imwrite(I,map,'SVM_RBF_xor.gif','gif','WriteMode','append','DelayTime',0.2);
    end
    close all
end