clear;
clc;
%%
%��һ����Ƭ��ʵ�飬�ǳɹ���
%��chip5 c2ʵ��ɹ���
row = 700;col = 600;
filepath = 'D:\ѧϰ\����\�Ӿ�ע��\��ʵ��\TestArea\chip6';%%����·��
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
% % figure,imshow(image_dct);  %%%ͼ��DCT�任�󣬵�Ƶ��Ϣ�����ھ�������Ͻǣ���Ƶ��Ϣ�������½Ǽ��С�
%                            %%%ֱ��������[0,0]����[0,1]���Ļ�������һ����������һ�������ڵ����Һ���������һ����������һ��������[1,0]���Ļ�������[0,1]���ƣ�ֻ����������ת��90�ȡ�
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
des = 'D:\ѧϰ\����\�Ӿ�ע��\��ʵ��\TestArea\chip4';
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

filepath='D:\ѧϰ\����\2016����PCTģ��\TestArea\chip4';
row = 677;col = 887;
ThreeComponent = {};
ThreeComponent = mchidecomposition(filepath,row,col);
[H,a] = Decomposition_Lin(filepath,row,col);

% resulta= PCTmodel(ThreeComponent{2},3,1.5);%��ɢ�����
% de_sidelobe =ThreeComponent{2}.*cos(a);
% resultb= PCTmodel(de_sidelobe,3,1.5);%��ɢ��*a���Ӿ�ע�⣬����

de_falsealarm = ThreeComponent{2}.*cos(ThreeComponent{5}/2);
saliency= PCTmodel(de_falsealarm,3,1.5); %��ɢ��*delta���Ӿ�ע�⣬����
modifiedsaliency = saliency.*cos(abs(a));
figure,imshow(modifiedsaliency);
% de_all = ThreeComponent{2}.*abs(sin(ThreeComponent{5})).*cos(a);
% resultd = PCTmodel(de_all,2,1.5);%��ɢ��*delta*a���Ӿ�ע�⣬����
% resultd = PCTmodel(de_all,4,1.5);%��ɢ��*delta*a���Ӿ�ע�⣬����

%������΢����

%��ֲ�
%ѡ��������
oceanarea = modifiedsaliency(417:634,244:696);
figure,imshow(modifiedsaliency);
hold on;
rectangle('Position',[244,417,453,218],'EdgeColor','r');
hold off;
figure;
[n1,ImagOut]=hist(oceanarea(:),1000);         %��Ϊ100������ͳ�ƣ�(����Ը�����Ҫ��������)
n1 = n1/sum(n1);
figure,plot(ImagOut,n1,'.','MarkerSize',5);     
hold on;
%�ں��������ڣ��ֱ���ϸ�˹�ֲ���gamma�ֲ���ָ���ֲ��������ֲ������KL���룬�������ͼ��

%%
%��˹�ֲ�����
mu = mean(oceanarea(:));
msigma = std(oceanarea(:)); %�����Ȼ�������
F = @(x)exp(-(x-mu).^2/(2*msigma^2))/(sqrt(2*pi)*msigma); %��˹�ֲ������ܶȺ���
GaussianPdf = zeros([1 1000]);
start1 = min((min(oceanarea)));
GaussianPdf(1) = quad(F,start1,ImagOut(1));
for i=2:1000
    GaussianPdf(i) = quad(F,ImagOut(i-1),ImagOut(i));
end
%KL Testt
D1_gaussian = 0;
D2_gaussian = 0;
n1 = n1+eps; %��ֹ����0ֵ
GaussianPdf = GaussianPdf+eps; %��ֹ����0ֵ
for i=1:1000
    D1_gaussian = D1_gaussian+n1(i)*log2(n1(i)/GaussianPdf(i));
    D2_gaussian = D2_gaussian+GaussianPdf(i)*log2(GaussianPdf(i)/n1(i));
end
D_gaussian = D1_gaussian+D2_gaussian;

%KS����
K_gaussian = max(abs(GaussianPdf-n1));
%MSE
RMSE_gaussian = sqrt(mean((GaussianPdf-n1).^2));
%%
%��������
msigma = mean2(oceanarea.^2); %�����Ȼ�������(��P85)
F = @(x)2*x.*exp(-x.^2/msigma)/msigma; %��˹�ֲ������ܶȺ���
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
%KS����
K_rayleigh = max(abs(RayleighPdf-n1));
%RMSE

RMSE_rayleigh = sqrt(mean((RayleighPdf-n1).^2));
%%
%ָ���ֲ�
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
%KS����
K_exp = max(abs(ExpPdf-n1));
%RMSE

RMSE_exp = sqrt(mean((ExpPdf-n1).^2));
%%
%gamma�ֲ�
%ţ������ɭ�㷨
%û��ʵ��
PARMHAT = gamfit(oceanarea(:)) ;
Y = zeros([1 1000]);
Y = gampdf(n1,PARMHAT(1),PARMHAT(2));

%%
%������̬
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
%KS����
K_lognormal = max(abs(Lognormalpdf-n1));
%RMSE

RMSE_lognormal = sqrt(mean((Lognormalpdf-n1).^2));
%%
%�ֲ�ͼ
figure,plot(ImagOut,n1,'.','MarkerSize',5);
hold on
plot(ImagOut,RayleighPdf,'r','LineWidth',2);
 plot(ImagOut,GaussianPdf,'g','LineWidth',2);
 plot(ImagOut,ExpPdf,'k','LineWidth',2);
plot(ImagOut,Lognormalpdf,'m','LineWidth',2);


%******************* chip7 ********************************
clc;
clear;
filepath='D:\ѧϰ\����\�Ӿ�ע��\��ʵ��\TestArea\chip7';
row = 2000;
col = 1300;
ThreeComponent = {};
ThreeComponent = mchidecomposition(filepath,row,col);
[H,a] = Decomposition_Lin(filepath,row,col);

de_falsealarm = ThreeComponent{2}.*abs(sin(ThreeComponent{5}));
saliency= PCTmodel(de_falsealarm,3,1.5); %��ɢ��*delta���Ӿ�ע�⣬����
modifiedsaliency = saliency.*cos(abs(a));

%��׼��̬�ۻ����ʺ���phai��x��
%phai(x)=1-pdf=0.995ʱ��x=2.58(����)
%phai(x)=1-pdf=0.9999ʱ��x=3.08
logocean = log(modifiedsaliency);
mu = mean2(modifiedsaliency);
msigma =std(modifiedsaliency(:));
T = msigma*3.08+mu;
result1 = modifiedsaliency>T;
figure,imshow(result1);title('PCT�����');
%significance
background = modifiedsaliency<T;
background = background.*modifiedsaliency;
target = result1.*modifiedsaliency;
sig1 = (max(target(:))-mean(background(:)))/std(background(:));



%******************* chip8 ********************************
clc;
clear;
filepath='D:\ѧϰ\����\�Ӿ�ע��\��ʵ��\TestArea\chip8';
row = 1500;
col = 1500;
ThreeComponent = {};
ThreeComponent = mchidecomposition(filepath,row,col);
[H,a] = Decomposition_Lin(filepath,row,col);

de_falsealarm = ThreeComponent{2}.*abs(sin(ThreeComponent{5}));
saliency= PCTmodel(de_falsealarm,3,1.5); %��ɢ��*delta���Ӿ�ע�⣬����
modifiedsaliency = saliency.*cos(abs(a));
figure,imshow(modifiedsaliency);title('������ͼ');
% modifiedsaliency2 = saliency.*cos(a).*cos(a);
% figure,imshow(modifiedsaliency2);
[y,x]=hist(modifiedsaliency,[-0.02:0.03:2]);         %��Ϊ100������ͳ�ƣ�(����Ը�����Ҫ��������)
y = sum(y,2);
y=y/(size(modifiedsaliency,1)*size(modifiedsaliency,2));   %��������ܶ� ��Ƶ�����������������������
%cftool
%a = 0.1172;
%b = -9.674
T=-b*log(pfa);
result1 = modifiedsaliency>1.7057;
figure,imshow(result1);title('PCT�����');
%significance
background = find(modifiedsaliency<1.7057);
background = background.*modifiedsaliency;
target = result1.*modifiedsaliency;
sig2 = (max(target(:))-mean(background(:)))/std(background(:));
