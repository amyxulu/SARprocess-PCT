clc;
clear;
%%
%研究示例区域
filepath='D:\学习\程序\视觉注意\新实验\TestArea\chip4';
row = 677;col = 887;
ThreeComponent = {};
ThreeComponent = mchidecomposition(filepath,row,col);
[H,a] = Decomposition_Lin(filepath,row,col);
de_falsealarm = ThreeComponent{2}.*cos(ThreeComponent{5}/2);
saliency= PCTmodel(de_falsealarm); %体散射*delta再视觉注意，可行
modifiedsaliency = saliency.*cos(abs(a));
figure,imshow(modifiedsaliency);

%********************************************************************************************************************************
%选择海洋背景
oceanarea = modifiedsaliency(417:634,244:696);
figure,imshow(modifiedsaliency);
hold on;
rectangle('Position',[244,417,453,218],'EdgeColor','r');
hold off;

%********************************************************************************************************************************
%统计直方图
[n1,ImagOut]=hist(oceanarea(:),1000);         %分为100个区间统计，(你可以改你需要的区间数)
n1 = n1/sum(n1);
%在海洋区域内，分别拟合高斯分布，gamma分布，指数分布，瑞利分布，求出KL距离，画出拟合图像

%********************************************************************************************************************************
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
%KL 检验
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

%********************************************************************************************************************************
%瑞利检验
msigma = mean2(oceanarea.^2); %最大似然法求参数(书P85)
F = @(x)2*x.*exp(-x.^2/msigma)/msigma; %高斯分布概率密度函数
RayleighPdf = zeros([1 1000]);
start1 = min((min(oceanarea)));
RayleighPdf(1) = quad(F,start1,ImagOut(1));
for i=2:1000
    RayleighPdf(i) = quad(F,ImagOut(i-1),ImagOut(i));
end
%KL 检验
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

%********************************************************************************************************************************
%指数分布
mu = mean(oceanarea(:));
F = @(x)1/mu*exp(-x/mu);
ExpPdf = zeros([1 1000]);
start1 = min((min(oceanarea)));
ExpPdf(1) = quad(F,start1,ImagOut(1));
for i=2:1000
    ExpPdf(i) = quad(F,ImagOut(i-1),ImagOut(i));
end
%KL 检验
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

%********************************************************************************************************************************
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

%分布图
figure,plot(ImagOut,n1,'.','MarkerSize',5);
hold on
plot(ImagOut,RayleighPdf,'r','LineWidth',2);
plot(ImagOut,GaussianPdf,'g','LineWidth',2);
plot(ImagOut,ExpPdf,'k','LineWidth',2);
plot(ImagOut,Lognormalpdf,'m','LineWidth',2);
hold off

