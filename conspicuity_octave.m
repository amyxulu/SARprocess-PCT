%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%���ɸ�������������ͼ
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function result = conspicuity_octave(data)
[row col] = size(data);
pyr = {};
w=fspecial('gaussian',[3 3]);
pyr{1} = data;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%PART III:centre-surround differences and normalization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%step1:������˹������
%(1)��˹�˲�
%��˹�˺���
%function kernel = gaussian(peak,sigma,maxhw,varargin)
%function result = sepConv2PreserveEnergy(filter1,filter2,data)
%(2)���²���
% gaussSPAN = gaussianSubsample(SPAN);
%%%%%%%%%%%%
for i=2:9
    pyr{i}=imresize(imfilter(pyr{i-1},w),[row/(2^(i-1)) col/(2^(i-1))]);
    %����
    %pyr2{i}=imresize(imfilter(pyr{i-1},w),0.5);
    %�����Ľ����һ�����������
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%step2:��������֣�ÿ��һ��ͼ��)
%���֮ǰ����imresize����
%��ֽ���ۼ�
%%%%%%%%%%%%
%siz = size(pyr.levels(lp.mapLevel).data);
%��һ�䲻����ʵ
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
        %c_s{i-1} = attenuateBorders(abs(pyr_resize{i}-pyr_resize{i+j}),bordersize);%%%%%%%%%%%%%%��һ��Ӧ������Щ���󣿣���
        c_s{k} = attenuateBorders(abs(pyr_resize{i}-pyr_resize{i+j}),bordersize);
        %c_s{k} = abs(pyr_resize{i}-pyr_resize{i+j});
        k = k+1;
        %c_s2{i} = abs(pyr_resize{i}-pyr_resize{i+j});
        %c_s��c_s2�Ĳ�������ȫͼ�ı߿򲿷֡����ǲ�֪��Ϊɶ�߿������
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%PART IV:across-scale combinations and normalization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%a_c = zeros(size(c_s{1}));
c_s_norm = {};
minmax = [0,1];
for i = 1:k-1
        c_s_norm{i} =  maxNormalizeLocalMax(c_s{i},minmax);%һ��Ҫ��һ����Ϊʲô��һ���к�û�н��һ����
        %a_c = a_c + c_s_norm{i};
        %a_c = a_c+c_s{i};
end
result = c_s_norm{1}+c_s_norm{2}+c_s_norm{3}+c_s_norm{4}+c_s_norm{5}+c_s_norm{6}; 
