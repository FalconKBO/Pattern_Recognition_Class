%E:\OnlineDisk\OneDrive\OneDrive - Neuroinformatics Collaboratory\Github\Pattern Recognition\SVM
stepall=40;%图像总帧数
for i=1:stepall
    picname=[num2str(i) '.fig'];
    open(picname)
     set(gcf,'outerposition',get(0,'screensize'));% matlab窗口最大化
    frame=getframe(gcf);  
    im=frame2im(frame);%制作gif文件，图像必须是index索引图像  
    %[I,map]=rgb2ind(im,20);  
    [I,map]=rgb2ind(im,256,'nodither');  %可以查看rbg2ind选择需要的转化格式和索引位数
    if i==1
        imwrite(I,map,'SVM_RBF_xor.gif','gif', 'Loopcount',inf,'DelayTime',0.2);%第一次必须创建！
    elseif i==stepall
        imwrite(I,map,'SVM_RBF_xor.gif','gif','WriteMode','append','DelayTime',0.2);
    else
        imwrite(I,map,'SVM_RBF_xor.gif','gif','WriteMode','append','DelayTime',0.2);
    end
    close all
end