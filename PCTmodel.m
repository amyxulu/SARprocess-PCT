function [result] = PCTmodel(image)
%n是降采样的倍数，m是DCT模板大小的倍数
[row,col] = size(image);
% image2 = imresize(image,[floor(row/n) floor(col/n)],'bilinear');
% dctsize = floor(m*max(size(image2)));
% T1 = dctmtx(dctsize);
% dct = @(x)T1 * x *T1';
% invdct = @(x)T1'*x*T1;
% image_dct = blkproc(image2,[dctsize dctsize],dct);
% % figure,imshow(image_dct);  %%%图像经DCT变换后，低频信息集中在矩阵的左上角，高频信息则向右下角集中。
%                            %%%直流分量在[0,0]处，[0,1]处的基函数在一个方向上是一个半周期的余弦函数，在另一个方向上是一个常数。[1,0]处的基函数与[0,1]类似，只不过方向旋转了90度。
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
%不用降采样
F(F<0) = 0;
F2 = F.^2;
% figure,imshow(F);
gaussKernal = fspecial('gaussian',[10 10],1);  
M = imfilter(F2,gaussKernal,'symmetric','same','conv');
% figure,imshow(M);
M = imresize(M,[row col],'bilinear');
% figure,imshow(M);
result = M;
