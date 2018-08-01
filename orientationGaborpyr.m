function result = orientationGaborpyr(imGray,theta,sigma)

%******************生成Gabor模板******************%  
lambda = 2;
phi = 0;
gamma = 1;  
%%theta是变化的角度
theta = theta*pi/180;%弧度
%理论上应该使用带宽b和波长lambda计算sigma的值，这里为了便于调试，直接定义sigma的值 
%%%%%sigma就是原文中的标准差，Itti说标准差在[0..8]之间
M = 7;N = 7;R = 10;  
%%高斯窗口的大小
[X,Y] = meshgrid(linspace(-R,R,N),linspace(-R,R,M));  
X1 = X .* cos(theta) + Y .* sin(theta);  
Y1 = Y .* cos(theta) + X .* sin(theta);  
%%生成X,Y坐标
gaborKernal = exp(- (X1.^2 + gamma^2 .* Y1.^2) ./ (2 * sigma^2)) .* cos(2 .* pi .* X1 ./lambda + phi);  

%******************标准化Gabor模板******************%  
gaborKernalSum = sum(gaborKernal(:));  
gaborKernalSum = gaborKernalSum / (M * N);  
gaborNormalKernal = gaborKernal - gaborKernalSum;  
 %******************生成L层高斯金字塔******************%  
% L = 5;  %定义金字塔层数  
% gaussPyrLevel = cell(L);  
% gaussKernal = fspecial('gaussian',[5 5],1);  
% gaussPyrLevel{1} = imfilter(imGray,gaussKernal,'symmetric','same','conv'); %高斯金字塔基图像（第0级图像）  
% rowSample = downsample(gaussPyrLevel{1},2);  
% columnSample = downsample(rowSample',2);  
% gaussPyrLevel{2} = columnSample';                                            
% for i = 3 : L  
%     levelTemp = imfilter(gaussPyrLevel{i - 1},gaussKernal,'symmetric','same','conv');  
%     rowSample = downsample(levelTemp,2);  
%     columnSample = downsample(rowSample',2);  
%     gaussPyrLevel{i} = columnSample';  
% end  

  %******************生成L层平均金字塔******************%  
L = 9;  
avePyrLevel = {};  
avePyrLevel{1} = imGray;  
for k = 2 : L  
    [m,n] = size(avePyrLevel{k - 1});  
    m = floor(m / 2);n = floor(n / 2);  
    avePyrLevel{k} = zeros(m,n);  
    for i = 1 : m                                                %计算大小为2*2的网格内的像素均值  
        for j = 1 : n  
            avePyrLevel{k}(i,j) = (avePyrLevel{k - 1}(i * 2,j * 2) + avePyrLevel{k - 1}(i * 2,j * 2 - 1) ...  
                + avePyrLevel{k - 1}(i * 2 - 1,j * 2) + avePyrLevel{k - 1}(i * 2 - 1,j * 2 - 1)) / 4;  
        end  
    end  
end 
%******************生成L层Gabor金字塔******************%  
gaborPyrLevel = {};  
for i = 1 : L  
    gaborPyrLevel{i} = imfilter(avePyrLevel{i},gaborNormalKernal,'symmetric','same','conv');  
end  
result ={};
result = gaborPyrLevel;

%%%%接下来应该是center-surround过程
% 把所有的特征resize，然后相减，然后attendborder
 siz = size(imGray);
 %[row col] = size(imGray);
pyr_resize={};
for i=1:L
    pyr_resize{i}= imresize(gaborPyrLevel{i}, siz,'bilinear');
end

c_s = {};
bordersize = round(max(siz)/20);
k=1;
for i=2:4
    for j = 3:4
        %c_s{i-1} = attenuateBorders(abs(pyr_resize{i}-pyr_resize{i+j}),bordersize);%%%%%%%%%%%%%%这一句应该是有些错误？？？
        c_s{k} = attenuateBorders(abs(pyr_resize{i}-pyr_resize{i+j}),bordersize);
        k = k+1;
        %c_s2{i} = abs(pyr_resize{i}-pyr_resize{i+j});
        %c_s与c_s2的差别表现在全图的边框部分。但是不知道为啥边框会这样
    end
end
result=c_s;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%PART IV:across-scale combinations and normalization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%a_c = zeros(size(c_s{1}));
c_s_norm = {};
minmax = [0,10];
for i = 1:k-1
        c_s_norm{i} =  maxNormalizeLocalMax(c_s{i},minmax);%一定要归一化吗？为什么这一句有和没有结果一样？
        %a_c = a_c + c_s_norm{i};
        %a_c = a_c+c_s{i};
end
result = c_s_norm{1}+c_s_norm{2}+c_s_norm{3}+c_s_norm{4}+c_s_norm{5}+c_s_norm{6}; 
%%%%%%%%%%%此时，做完了across-scale combination。


% 
% %******************生成方向特征图******************%  
% gaborPyrLevelZoom = imresize(gaborPyrLevel{3},[imInfo.Height imInfo.Width],'bilinear');  
% dirFeature = abs(gaborPyrLevelZoom - gaborPyrLevel{1});  
%   
% %******************标准化方向特征图******************%  
% dirFeatureSum = sum(dirFeature(:));  
% dirFeatureSum = dirFeatureSum / (imInfo.Height * imInfo.Width);  
% dirFeature =dirFeature - dirFeatureSum;  
%   
% %******************将取值区间移到[0,1]内，方便显示******************%  
% valMax = max(gaborNormalKernal(:));  
% valMin = min(gaborNormalKernal(:));  
% gaborNormalKernal = (gaborNormalKernal - valMin) ./ (valMax - valMin);  
%   
% for i = 1 : 5  
%     valMax = max(gaborPyrLevel{i}(:));  
%     valMin = min(gaborPyrLevel{i}(:));  
%     gaborPyrLevel{i} = (gaborPyrLevel{i} - valMin) ./ (valMax - valMin);  
% end  
%   
% valMax = max(dirFeature(:));  
% valMin = min(dirFeature(:));  
% dirFeature = (dirFeature - valMin) ./ (valMax - valMin);  
%   