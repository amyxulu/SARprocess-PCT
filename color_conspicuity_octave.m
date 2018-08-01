%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%生成颜色特征的显著图
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function result = color_conspicuity_octave(R,G,B,Y)
[row col] = size(R);
pyrR = {};
pyrG = {};
pyrB = {};
pyrY = {};
w=fspecial('gaussian',[3 3]);
pyrR{1} = R;
pyrG{1} = G;
pyrB{1} = B;
pyrY{1} = Y;
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
    pyrR{i}=imresize(imfilter(pyrR{i-1},w),[row/(2^(i-1)) col/(2^(i-1))]);
    pyrG{i}=imresize(imfilter(pyrG{i-1},w),[row/(2^(i-1)) col/(2^(i-1))]);
    pyrB{i}=imresize(imfilter(pyrB{i-1},w),[row/(2^(i-1)) col/(2^(i-1))]);
    pyrY{i}=imresize(imfilter(pyrY{i-1},w),[row/(2^(i-1)) col/(2^(i-1))]);
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
siz = size(R);
pyrR_resize={};
pyrG_resize={};
pyrB_resize={};
pyrY_resize={};
for i=1:9
    pyrR_resize{i}= imresize(pyrR{i},[row col] ,'bilinear');
    pyrG_resize{i}= imresize(pyrG{i},[row col] ,'bilinear');
    pyrB_resize{i}= imresize(pyrB{i},[row col] ,'bilinear');
    pyrY_resize{i}= imresize(pyrY{i},[row col] ,'bilinear');
end

c_sRG = {};
c_sBY = {};
k=1;
bordersize = round(max(siz)/20);
for i=2:4
    for j = 3:4
        c_sRG{k} = attenuateBorders(abs((pyrR_resize{i}-pyrG_resize{i})-(pyrG_resize{i+j}-pyrR_resize)),bordersize);
        c_sBY{k} = attenuateBorders(abs((pyrB_resize{i}-pyrY_resize{i})-(pyrY_resize{i+j}-pyrB_resize)),bordersize);
        k = k+1;
        %c_s2{i} = abs(pyr_resize{i}-pyr_resize{i+j});
        %c_s与c_s2的差别表现在全图的边框部分。但是不知道为啥边框会这样
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%PART IV:across-scale combinations and normalization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%a_c = zeros(size(c_s{1}));
c_s_normRG = {};
c_s_normBY = {};
minmax = [0,10];
for i = 1:k
        c_s_normRG{i} =  maxNormalizeLocalMax(c_sRG{i},minmax);%一定要归一化吗？为什么这一句有和没有结果一样？
        c_s_normBY{i} =  maxNormalizeLocalMax(c_sBY{i},minmax);
        result = result + c_s_normRG{i}+c_s_normBY{i};
        %a_c = a_c + c_s_norm{i};
        %a_c = a_c+c_s{i};
end
%result = c_s_normRG{1}+c_s_normRG{2}+c_s_normRG{3}+c_s_normBY{1}+c_s_normBY{2}+c_s_normBY{3}; 
