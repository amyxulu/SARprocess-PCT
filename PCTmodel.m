function [result] = PCTmodel(image)
%n�ǽ������ı�����m��DCTģ���С�ı���
[row,col] = size(image);
% image2 = imresize(image,[floor(row/n) floor(col/n)],'bilinear');
% dctsize = floor(m*max(size(image2)));
% T1 = dctmtx(dctsize);
% dct = @(x)T1 * x *T1';
% invdct = @(x)T1'*x*T1;
% image_dct = blkproc(image2,[dctsize dctsize],dct);
% % figure,imshow(image_dct);  %%%ͼ��DCT�任�󣬵�Ƶ��Ϣ�����ھ�������Ͻǣ���Ƶ��Ϣ�������½Ǽ��С�
%                            %%%ֱ��������[0,0]����[0,1]���Ļ�������һ����������һ�������ڵ����Һ���������һ����������һ��������[1,0]���Ļ�������[0,1]���ƣ�ֻ����������ת��90�ȡ�
% image_idct = blkproc(image_dct,[dctsize dctsize], invdct);
% % figure,imshow(image_idct);
% P = sign(image_dct);
% % figure,imshow(P);
% F= blkproc(P,[dctsize dctsize], invdct);
% figure,imshow(F);
image_dct = dct2(image);
x = idct2(image_dct);
P = sign(image_dct);
F = idct2(P);
%���ý�����
F(F<0) = 0;
F2 = F.^2;
% figure,imshow(F);
gaussKernal = fspecial('gaussian',[10 10],1);  
M = imfilter(F2,gaussKernal,'symmetric','same','conv');
% figure,imshow(M);
M = imresize(M,[row col],'bilinear');
% figure,imshow(M);
result = M;
