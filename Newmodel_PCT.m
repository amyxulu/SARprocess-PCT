clear;
clc;
%%
%用一般照片做实验，是成功的
%用chip5 c2实验成功了
row = 700;col = 600;
filepath = 'D:\学习\程序\视觉注意\新实验\TestArea\chip6';%%定义路径
image2 = readfile([filepath '\0713-1span'],row,col,1);
% n = 8;
% image3 = imresize(image2,[floor(row/n) floor(col/n)],'bilinear');
% figure,imshow(image2);
% dctsize = floor(0.5*max(size(image3)));
% T1 = dctmtx(dctsize);
% dct = @(x)T1 * x *T1';
% invdct = @(x)T1'*x*T1;
% 
% image_dct = blkproc(image3,[dctsize dctsize],dct);
% % figure,imshow(image_dct);  %%%图像经DCT变换后，低频信息集中在矩阵的左上角，高频信息则向右下角集中。
%                            %%%直流分量在[0,0]处，[0,1]处的基函数在一个方向上是一个半周期的余弦函数，在另一个方向上是一个常数。[1,0]处的基函数与[0,1]类似，只不过方向旋转了90度。
% image_idct = blkproc(image_dct,[dctsize dctsize], invdct);
% P = sign(image_dct);
% F= blkproc(P,[dctsize dctsize], invdct);

image_dct = dct2(image2);
P = sign(image_dct2);
F = idct2(P);
% x = idct2(image_dct);
F(F<0) = 0;
F2 = F.^2;
gaussKernal = fspecial('gaussian',[10 10],1);  
M = imfilter(F2,gaussKernal,'symmetric','same','conv');
M = imresize(M,[row col],'bilinear');
figure,imshow(M);
figure,imshow(M.*image2);
% writeToRmg('D:\PCTsalient',M,row,col,1);
%%
clc;
clear;
des = 'D:\学习\程序\视觉注意\新实验\TestArea\chip4';
row = 677;col = 887;
s11 = readfile([des '\s11.bin'],row,col,2);
s12 = readfile([des '\s12.bin'],row,col,2);
s21 = readfile([des '\s21.bin'],row,col,2);
s22 = readfile([des '\s22.bin'],row,col,2);

CTLR1 = s11-1i*s12;
CTLR2 = s12-1i*s22;
Amplitude1 = abs(CTLR1);
Amplitude2 = abs(CTLR2);
figure;imshow(Amplitude1,[0,0.1]);title('Intensity of CTLR 1');  %%%  RH
figure;imshow(Amplitude2,[0,0.1]);title('Intensity of CTLR 2');  %%%  RV
figure;imshow(CTLR1,[0,0.1]);title('CTLR 1');  %%%  RH
% figure;imshow(CTLR2,[0,0.1]);title('CTLR 2');  %%%  RV

% result1_1= PCTmodel(CTLR1,1,1);
% result1_2= PCTmodel(CTLR1,1,2);
% result2_1= PCTmodel(CTLR1,2,1);
% result2_2= PCTmodel(CTLR1,2,2);

result1= PCTmodel(Amplitude1,3,1.5);
result2= PCTmodel(Amplitude2,3,1.5);

filepath='D:\学习\程序\2016简缩PCT模型\TestArea\chip4';
row = 677;col = 887;
ThreeComponent = {};
ThreeComponent = mchidecomposition(filepath,row,col);
[H,a] = Decomposition_Lin(filepath,row,col);

% resulta= PCTmodel(ThreeComponent{2},3,1.5);%体散射分量
% de_sidelobe =ThreeComponent{2}.*cos(a);
% resultb= PCTmodel(de_sidelobe,3,1.5);%体散射*a再视觉注意，可行

de_falsealarm = ThreeComponent{2}.*cos(ThreeComponent{5}/2);
saliency= PCTmodel(de_falsealarm,3,1.5); %体散射*delta再视觉注意，可行
modifiedsaliency = saliency.*cos(abs(a));
figure,imshow(modifiedsaliency);
% de_all = ThreeComponent{2}.*abs(sin(ThreeComponent{5})).*cos(a);
% resultd = PCTmodel(de_all,2,1.5);%体散射*delta*a再视觉注意，可行
% resultd = PCTmodel(de_all,4,1.5);%体散射*delta*a再视觉注意，可行

%三种有微妙差别

%求分布
%选择海洋区域：
oceanarea = modifiedsaliency(417:634,244:696);
figure,imshow(modifiedsaliency);
hold on;
rectangle('Position',[244,417,453,218],'EdgeColor','r');
hold off;
figure;
[n1,ImagOut]=hist(oceanarea(:),1000);         %分为100个区间统计，(你可以改你需要的区间数)
n1 = n1/sum(n1);
figure,plot(ImagOut,n1,'.','MarkerSize',5);     
hold on;
%在海洋区域内，分别拟合高斯分布，gamma分布，指数分布，瑞利分布，求出KL距离，画出拟合图像

%%
%高斯分布检验
mu = mean(oceanarea(:));
msigma = std(oceanarea(:)); %最大似然法求参数
F = @(x)exp(-(x-mu).^2/(2*msigma^2))/(sqrt(2*pi)*msigma); %高斯分布概率密度函数
GaussianPdf = zeros([1 1000]);
start1 = min((min(oceanarea)));
GaussianPdf(1) = quad(F,start1,ImagOut(1));
for i=2:1000
    GaussianPdf(i) = quad(F,ImagOut(i-1),ImagOut(i));
end
%KL Testt
D1_gaussian = 0;
D2_gaussian = 0;
n1 = n1+eps; %防止出现0值
GaussianPdf = GaussianPdf+eps; %防止出现0值
for i=1:1000
    D1_gaussian = D1_gaussian+n1(i)*log2(n1(i)/GaussianPdf(i));
    D2_gaussian = D2_gaussian+GaussianPdf(i)*log2(GaussianPdf(i)/n1(i));
end
D_gaussian = D1_gaussian+D2_gaussian;

%KS检验
K_gaussian = max(abs(GaussianPdf-n1));
%MSE
RMSE_gaussian = sqrt(mean((GaussianPdf-n1).^2));
%%
%瑞利检验
msigma = mean2(oceanarea.^2); %最大似然法求参数(书P85)
F = @(x)2*x.*exp(-x.^2/msigma)/msigma; %高斯分布概率密度函数
RayleighPdf = zeros([1 1000]);
start1 = min((min(oceanarea)));
RayleighPdf(1) = quad(F,start1,ImagOut(1));
for i=2:1000
    RayleighPdf(i) = quad(F,ImagOut(i-1),ImagOut(i));
end

D1_rayleigh = 0; 
D2_rayleigh = 0; 
n1 = n1+eps;
RayleighPdf = RayleighPdf+eps;
for i=1:1000
    D1_rayleigh = D1_rayleigh+n1(i)*log2(n1(i)/RayleighPdf(i));
    D2_rayleigh = D2_rayleigh+RayleighPdf(i)*log2(RayleighPdf(i)/n1(i));
end
D_rayleigh = D1_rayleigh+D2_rayleigh;
%KS检验
K_rayleigh = max(abs(RayleighPdf-n1));
%RMSE

RMSE_rayleigh = sqrt(mean((RayleighPdf-n1).^2));
%%
%指数分布
mu = mean(oceanarea(:));
F = @(x)1/mu*exp(-x/mu);
ExpPdf = zeros([1 1000]);
start1 = min((min(oceanarea)));
ExpPdf(1) = quad(F,start1,ImagOut(1));
for i=2:1000
    ExpPdf(i) = quad(F,ImagOut(i-1),ImagOut(i));
end

D1_exp = 0; 
D2_exp = 0; 
n1 = n1+eps;
ExpPdf = ExpPdf+eps;

for i=1:1000
    D1_exp = D1_exp+n1(i)*log2(n1(i)/ExpPdf(i));
    D2_exp = D2_exp+ExpPdf(i)*log2(ExpPdf(i)/n1(i));
end
D_exp = D1_exp+D2_exp;
%KS检验
K_exp = max(abs(ExpPdf-n1));
%RMSE

RMSE_exp = sqrt(mean((ExpPdf-n1).^2));
%%
%gamma分布
%牛顿拉夫森算法
%没有实验
PARMHAT = gamfit(oceanarea(:)) ;
Y = zeros([1 1000]);
Y = gampdf(n1,PARMHAT(1),PARMHAT(2));

%%
%对数正态
logocean = log(oceanarea);
mu = mean2(logocean);
msigma =std(logocean(:));
F = @(x)exp(-(log(x)-mu).^2/(2*msigma^2))/(sqrt(2*pi).*msigma)./x;
Lognormalpdf = zeros([1 1000]);
start1 = min((min(oceanarea)));
Lognormalpdf(1) = quad(F,start1,ImagOut(1));
for i=2:1000
    Lognormalpdf(i) = quad(F,ImagOut(i-1),ImagOut(i));
end

D1_lognormal = 0; 
D2_lognormal = 0; 
n1 = n1+eps;
Lognormalpdf = Lognormalpdf+eps;
for i=1:1000
    D1_lognormal = D1_lognormal+n1(i)*log2(n1(i)/Lognormalpdf(i));
    D2_lognormal = D2_lognormal+Lognormalpdf(i)*log2(Lognormalpdf(i)/n1(i));
end
D_lognormal = D1_lognormal+D2_lognormal;
%KS检验
K_lognormal = max(abs(Lognormalpdf-n1));
%RMSE

RMSE_lognormal = sqrt(mean((Lognormalpdf-n1).^2));
%%
%分布图
figure,plot(ImagOut,n1,'.','MarkerSize',5);
hold on
plot(ImagOut,RayleighPdf,'r','LineWidth',2);
 plot(ImagOut,GaussianPdf,'g','LineWidth',2);
 plot(ImagOut,ExpPdf,'k','LineWidth',2);
plot(ImagOut,Lognormalpdf,'m','LineWidth',2);


%******************* chip7 ********************************
clc;
clear;
filepath='D:\学习\程序\视觉注意\新实验\TestArea\chip7';
row = 2000;
col = 1300;
ThreeComponent = {};
ThreeComponent = mchidecomposition(filepath,row,col);
[H,a] = Decomposition_Lin(filepath,row,col);

de_falsealarm = ThreeComponent{2}.*abs(sin(ThreeComponent{5}));
saliency= PCTmodel(de_falsealarm,3,1.5); %体散射*delta再视觉注意，可行
modifiedsaliency = saliency.*cos(abs(a));

%标准正态累积概率函数phai（x）
%phai(x)=1-pdf=0.995时，x=2.58(查表得)
%phai(x)=1-pdf=0.9999时，x=3.08
logocean = log(modifiedsaliency);
mu = mean2(modifiedsaliency);
msigma =std(modifiedsaliency(:));
T = msigma*3.08+mu;
result1 = modifiedsaliency>T;
figure,imshow(result1);title('PCT检测结果');
%significance
background = modifiedsaliency<T;
background = background.*modifiedsaliency;
target = result1.*modifiedsaliency;
sig1 = (max(target(:))-mean(background(:)))/std(background(:));



%******************* chip8 ********************************
clc;
clear;
filepath='D:\学习\程序\视觉注意\新实验\TestArea\chip8';
row = 1500;
col = 1500;
ThreeComponent = {};
ThreeComponent = mchidecomposition(filepath,row,col);
[H,a] = Decomposition_Lin(filepath,row,col);

de_falsealarm = ThreeComponent{2}.*abs(sin(ThreeComponent{5}));
saliency= PCTmodel(de_falsealarm,3,1.5); %体散射*delta再视觉注意，可行
modifiedsaliency = saliency.*cos(abs(a));
figure,imshow(modifiedsaliency);title('显著性图');
% modifiedsaliency2 = saliency.*cos(a).*cos(a);
% figure,imshow(modifiedsaliency2);
[y,x]=hist(modifiedsaliency,[-0.02:0.03:2]);         %分为100个区间统计，(你可以改你需要的区间数)
y = sum(y,2);
y=y/(size(modifiedsaliency,1)*size(modifiedsaliency,2));   %计算概率密度 ，频数除以数据种数，除以组距
%cftool
%a = 0.1172;
%b = -9.674
T=-b*log(pfa);
result1 = modifiedsaliency>1.7057;
figure,imshow(result1);title('PCT检测结果');
%significance
background = find(modifiedsaliency<1.7057);
background = background.*modifiedsaliency;
target = result1.*modifiedsaliency;
sig2 = (max(target(:))-mean(background(:)))/std(background(:));
