%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%生成各个特征的显著图
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function result = conspicuity_octave(data)
[row col] = size(data);
pyr = {};
w=fspecial('gaussian',[3 3]);
pyr{1} = data;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%PART III:centre-surround differences and normalization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%step1:制作高斯金字塔
%(1)高斯滤波
%高斯核函数
%function kernel = gaussian(peak,sigma,maxhw,varargin)
%function result = sepConv2PreserveEnergy(filter1,filter2,data)
%(2)向下采样
% gaussSPAN = gaussianSubsample(SPAN);
%%%%%%%%%%%%
for i=2:9
    pyr{i}=imresize(imfilter(pyr{i-1},w),[row/(2^(i-1)) col/(2^(i-1))]);
    %或者
    %pyr2{i}=imresize(imfilter(pyr{i-1},w),0.5);
    %两个的结果不一样啊，真奇怪
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%step2:进行组间差分（每组一副图像)
%差分之前进行imresize处理
%差分结果累加
%%%%%%%%%%%%
%siz = size(pyr.levels(lp.mapLevel).data);
%这一句不懂其实
siz = size(data);
pyr_resize={};
for i=1:9
    pyr_resize{i}= imresize(pyr{i},[row col] ,'bilinear');
end

c_s = {};
bordersize = round(max(siz)/20);
k=1;
for i=2:4
    for j = 3:4
        %c_s{i-1} = attenuateBorders(abs(pyr_resize{i}-pyr_resize{i+j}),bordersize);%%%%%%%%%%%%%%这一句应该是有些错误？？？
        c_s{k} = attenuateBorders(abs(pyr_resize{i}-pyr_resize{i+j}),bordersize);
        %c_s{k} = abs(pyr_resize{i}-pyr_resize{i+j});
        k = k+1;
        %c_s2{i} = abs(pyr_resize{i}-pyr_resize{i+j});
        %c_s与c_s2的差别表现在全图的边框部分。但是不知道为啥边框会这样
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%PART IV:across-scale combinations and normalization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%a_c = zeros(size(c_s{1}));
c_s_norm = {};
minmax = [0,1];
for i = 1:k-1
        c_s_norm{i} =  maxNormalizeLocalMax(c_s{i},minmax);%一定要归一化吗？为什么这一句有和没有结果一样？
        %a_c = a_c + c_s_norm{i};
        %a_c = a_c+c_s{i};
end
result = c_s_norm{1}+c_s_norm{2}+c_s_norm{3}+c_s_norm{4}+c_s_norm{5}+c_s_norm{6}; 
