function result = orientationGaborpyr(imGray,theta,sigma)

%******************����Gaborģ��******************%  
lambda = 2;
phi = 0;
gamma = 1;  
%%theta�Ǳ仯�ĽǶ�
theta = theta*pi/180;%����
%������Ӧ��ʹ�ô���b�Ͳ���lambda����sigma��ֵ������Ϊ�˱��ڵ��ԣ�ֱ�Ӷ���sigma��ֵ 
%%%%%sigma����ԭ���еı�׼�Itti˵��׼����[0..8]֮��
M = 7;N = 7;R = 10;  
%%��˹���ڵĴ�С
[X,Y] = meshgrid(linspace(-R,R,N),linspace(-R,R,M));  
X1 = X .* cos(theta) + Y .* sin(theta);  
Y1 = Y .* cos(theta) + X .* sin(theta);  
%%����X,Y����
gaborKernal = exp(- (X1.^2 + gamma^2 .* Y1.^2) ./ (2 * sigma^2)) .* cos(2 .* pi .* X1 ./lambda + phi);  

%******************��׼��Gaborģ��******************%  
gaborKernalSum = sum(gaborKernal(:));  
gaborKernalSum = gaborKernalSum / (M * N);  
gaborNormalKernal = gaborKernal - gaborKernalSum;  
 %******************����L���˹������******************%  
% L = 5;  %�������������  
% gaussPyrLevel = cell(L);  
% gaussKernal = fspecial('gaussian',[5 5],1);  
% gaussPyrLevel{1} = imfilter(imGray,gaussKernal,'symmetric','same','conv'); %��˹��������ͼ�񣨵�0��ͼ��  
% rowSample = downsample(gaussPyrLevel{1},2);  
% columnSample = downsample(rowSample',2);  
% gaussPyrLevel{2} = columnSample';                                            
% for i = 3 : L  
%     levelTemp = imfilter(gaussPyrLevel{i - 1},gaussKernal,'symmetric','same','conv');  
%     rowSample = downsample(levelTemp,2);  
%     columnSample = downsample(rowSample',2);  
%     gaussPyrLevel{i} = columnSample';  
% end  

  %******************����L��ƽ��������******************%  
L = 9;  
avePyrLevel = {};  
avePyrLevel{1} = imGray;  
for k = 2 : L  
    [m,n] = size(avePyrLevel{k - 1});  
    m = floor(m / 2);n = floor(n / 2);  
    avePyrLevel{k} = zeros(m,n);  
    for i = 1 : m                                                %�����СΪ2*2�������ڵ����ؾ�ֵ  
        for j = 1 : n  
            avePyrLevel{k}(i,j) = (avePyrLevel{k - 1}(i * 2,j * 2) + avePyrLevel{k - 1}(i * 2,j * 2 - 1) ...  
                + avePyrLevel{k - 1}(i * 2 - 1,j * 2) + avePyrLevel{k - 1}(i * 2 - 1,j * 2 - 1)) / 4;  
        end  
    end  
end 
%******************����L��Gabor������******************%  
gaborPyrLevel = {};  
for i = 1 : L  
    gaborPyrLevel{i} = imfilter(avePyrLevel{i},gaborNormalKernal,'symmetric','same','conv');  
end  
result ={};
result = gaborPyrLevel;

%%%%������Ӧ����center-surround����
% �����е�����resize��Ȼ�������Ȼ��attendborder
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
        %c_s{i-1} = attenuateBorders(abs(pyr_resize{i}-pyr_resize{i+j}),bordersize);%%%%%%%%%%%%%%��һ��Ӧ������Щ���󣿣���
        c_s{k} = attenuateBorders(abs(pyr_resize{i}-pyr_resize{i+j}),bordersize);
        k = k+1;
        %c_s2{i} = abs(pyr_resize{i}-pyr_resize{i+j});
        %c_s��c_s2�Ĳ�������ȫͼ�ı߿򲿷֡����ǲ�֪��Ϊɶ�߿������
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
        c_s_norm{i} =  maxNormalizeLocalMax(c_s{i},minmax);%һ��Ҫ��һ����Ϊʲô��һ���к�û�н��һ����
        %a_c = a_c + c_s_norm{i};
        %a_c = a_c+c_s{i};
end
result = c_s_norm{1}+c_s_norm{2}+c_s_norm{3}+c_s_norm{4}+c_s_norm{5}+c_s_norm{6}; 
%%%%%%%%%%%��ʱ��������across-scale combination��


% 
% %******************���ɷ�������ͼ******************%  
% gaborPyrLevelZoom = imresize(gaborPyrLevel{3},[imInfo.Height imInfo.Width],'bilinear');  
% dirFeature = abs(gaborPyrLevelZoom - gaborPyrLevel{1});  
%   
% %******************��׼����������ͼ******************%  
% dirFeatureSum = sum(dirFeature(:));  
% dirFeatureSum = dirFeatureSum / (imInfo.Height * imInfo.Width);  
% dirFeature =dirFeature - dirFeatureSum;  
%   
% %******************��ȡֵ�����Ƶ�[0,1]�ڣ�������ʾ******************%  
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