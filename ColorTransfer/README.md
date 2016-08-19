# ColorTransfer #

***

## 引言 ##

颜色迁移实际上是要解决这么一个问题：基于图像A和图像B，合成一副新的图像C，使其同时具有A的颜色和B的形状等遗传信息，即图像B在不改变它自身所表达的形状信息的情况下，学习了图像A的整体颜色基调。

其中，图像A成为样例(Target)图像，图像B成为输入(Source)图像。

针对颜色迁移，做了一些基础方法的调研，本文针对一些方法进行总结，并延伸出一些迁移方法。

1. Color Transfer between Images
2. Color Transfer in Correlated Color Space
3. Color Transfer using Principal Component Analysis
4. Color Transfer by PowerMap(Part of Style transfer)

## 1. 颜色空间转换 ##


## 2.1 Color Transfer between Images ##

《Color Transfer between Images》可以看作颜色迁移技术的开山鼻祖。

Reinhard等人根据lαβ颜色空间中各通道互相不关联的特点，提出了一组适用于各颜色分量的色彩迁移公式，较好的实现了彩色图像之间的色彩迁移。基本思想就是根据着色图像的统计分析确定一个线性变换，使得目标图像和源图像在lαβ空间中具有同样的均值和方差。

因此需要计算两幅图像的均值和标准方差。假设l、a、b分别是源图像lαβ通道原有的数据，L、A、B分别是变换后得到新的源图像lαβ通道的值，ml、ma、mb和ml’、ma’、mb’分别是源图像和着色图像的三个颜色通道的均值，nl、na、nb和nl’、na’、nb’表示它们的标准方差。

首先，将源图像原有的数据减掉源图像的均值

	L = l – ml

	A = a – ma

	B = b – mb

再将得到的新数据按比例放缩，其放缩系数是两幅图像标准方差的比值

	L’ = (nl’ / nl)* L

	A’ = (na’ / na)* A

	B’ = (nb’ / nb)* B

将得到的l’、a’、b’分别加上目标图像三个通道的均值，得到最终数据

	L = L’ + ml’

	A = A’ + ma’

	B = B’ + mb’

整理后得到目标图像与源图像之间的像素关系表达

	L = (nl’ / nl)* (l – ml) + ml’

	A = (na’ / na)* (a – ma) + ma’

	B = (nb’ / nb)* (b – mb) + mb’

事实上这组式子表示的就是一个线性方程，以两幅图像的标准方差的比值作为斜率，两幅图像各个通道的均值作为一个点。这个简单的线性变换保证了目标图像和着色图像在lαβ颜色空间中具有相同的均值和方差。将变换后l、α、β的值作为新图像三个通道的值，然后显示的时候再将这三个通道转换为RGB值进行显示。


    %Reinhard颜色迁移算法
    function [dstImg] = colormatch(srcImg, tgtImg)

	[tgt_L,tgt_A,tgt_B] = RGB2lab(tgtImg);
	[src_L,src_A,src_B] = RGB2lab(srcImg);
    
	%  to impose the color of source(src) on the target(tgt) image
	dst_L = impose(src_L,tgt_L);           
	dst_A = impose(src_A,tgt_A);
	dst_B = impose(src_B,tgt_B);
	dstImg = lab2RGB(dst_L,dst_A,dst_B);
    
    end

	function [tp] = impose(src,tgt)
    %  to impose the color of source(src) on the target(tgt) image
    %  Written by Ruchir Srivastava

    td=std2(tgt);
    tm=mean2(tgt);
    sd=std2(src);
    sm=mean2(src);

    tp=(tgt-tm).*(sd/td)+sm;

    end

论文将RGB映射到Lab空间当中进行迁移计算，首次引发尝试使用YCbCr空间以及YIQ空间，发现都可以得到不错的效果。


## 2.2 Color Transfer in Correlated Color Space ##

《Color Transfer in Correlated Color Space》受ReinHard颜色迁移的启发，直接对任意的3D空间对颜色迁移进行处理。
论文的算法将每个像素视为一个三维随机变量，所以通过引入协方差可以表示三个成分的关联。然后，通过对协方差进行SVD分解，进而获得一个旋转矩阵。最后可以通过缩放、旋转和平移来进行颜色迁移。本文未考虑论文中Swatch-based transfer的情况。

图像的统计变量表示如下：
<img src="http://www.forkosh.com/mathtex.cgi? \bar{R}_{src}">表示输入图像R分量的均值，
<img src="http://www.forkosh.com/mathtex.cgi? \bar{G}_{src}">表示输入图像G分量的均值，
<img src="http://www.forkosh.com/mathtex.cgi? \bar{B}_{src}">表示输入图像B分量的均值，
<img src="http://www.forkosh.com/mathtex.cgi? \bar{R}_{tgt}">表示样例图像R分量的均值，
<img src="http://www.forkosh.com/mathtex.cgi? \bar{G}_{tgt}">表示样例图像G分量的均值，
<img src="http://www.forkosh.com/mathtex.cgi? \bar{B}_{tgt}">表示样例图像B分量的均值，
<img src="http://www.forkosh.com/mathtex.cgi? Cov_{src}">表示输入图像RGB的协方差矩阵，
<img src="http://www.forkosh.com/mathtex.cgi? Cov_{tgt}">表示样例图像RGB的协方差矩阵。

使用SVD分解表示协方差矩阵，公式如下：

<img src="http://www.forkosh.com/mathtex.cgi? Cov = U \cdot \Lambda \cdot V{^T} ">

<img src="http://www.forkosh.com/mathtex.cgi? \Lambda = diag(\lambda{^R}, \lambda{^G}, \lambda{^B}) ">，<img src="http://www.forkosh.com/mathtex.cgi? \lambda{^R}, \lambda{^G}, \lambda{^B} ">为Cov中的特征值。

颜色转换核心公式：

<img src="http://www.forkosh.com/mathtex.cgi? I = T_{src} \cdot R_{src} \cdot S_{src} \cdot S_{tgt} \cdot R_{tgt} \cdot T_{tgt} \cdot I_{tgt} ">

<img src="http://www.forkosh.com/mathtex.cgi? I = (R,G,B,1)^T, I_{tgt} = (R_{tgt},G_{tgt},B_{tgt},1)^T">

<img src="http://www.forkosh.com/mathtex.cgi? T_{src}, T_{tgt}, R_{src}, R_{tgt}, S_{src}, S_{tgt}"> 分别表示输入图像和样例图像的变换矩阵、旋转矩阵和缩放矩阵。

<img src="http://www.forkosh.com/mathtex.cgi? T_{src} = \begin{bmatrix}
1 & 0 & 0 & t_{src}^{r}\\ 
0 & 1 & 0 & t_{src}^{g}\\ 
0 & 0 & 1 & t_{src}^{b}\\ 
0 & 0 & 0 & 1
\end{bmatrix}, T_{tgt} = \begin{bmatrix}
1 & 0 & 0 & t_{tgt}^{r}\\ 
0 & 1 & 0 & t_{tgt}^{g}\\ 
0 & 0 & 1 & t_{tgt}^{b}\\ 
0 & 0 & 0 & 1
\end{bmatrix}">


<img src="http://www.forkosh.com/mathtex.cgi? R_{src} = U_{src}, R_{tgt} = U_{tgt}^{-1}">

<img src="http://www.forkosh.com/mathtex.cgi? S_{src} = \begin{bmatrix}
S_{src}^r & 0 & 0 & 0\\ 
0 & s_{src}^g & 0 & 0\\ 
0 & 0 & s_{src}^b & 0\\ 
0 & 0 & 0 & 1
\end{bmatrix}, S_{tgt} = \begin{bmatrix}
s_{tgt}^r & 0 & 0 & 0\\ 
0 & s_{tgt}^g & 0 & 0\\ 
0 & 0 & s_{tgt}^b & 0\\ 
0 & 0 & 0 & 1
\end{bmatrix}">

<img src="http://www.forkosh.com/mathtex.cgi? t_{src}^{r} = \bar{R}_{src}, t_{src}^{g} = \bar{G}_{src}, t_{src}^{b} = \bar{B}_{src}, s_{src}^{r} = \sqrt{\lambda_{src}^{R}}, s_{src}^{g} = \sqrt{\lambda_{src}^{G}}, s_{src}^{b} = \sqrt{\lambda_{src}^{B}}"> 

<img src="http://www.forkosh.com/mathtex.cgi?t_{tgt}^{r} = -\bar{R}_{tgt}, t_{tgt}^{g} = -\bar{G}_{tgt}, t_{tgt}^{b} = -\bar{B}_{tgt}, s_{tgt}^{r} = 1 / \sqrt{\lambda_{tgt}^{R} }, s_{tgt}^{g} = 1 / \sqrt{\lambda_{tgt}^{G} }, s_{tgt}^{b} = 1 / \sqrt{\lambda_{tgt}^{B} }"> 

	%Color Transfer in Correlated Color Space颜色迁移算法
    function dstImg = ColorTransfer_by_Correlated(srcImg,tgtImg)

	rgb_s = reshape(im2double(srcImg),[],3)';
	rgb_t = reshape(im2double(tgtImg),[],3)';

	% compute mean
	mean_s = mean(rgb_s,2);
	mean_t = mean(rgb_t,2);

	% compute covariance
	cov_s = cov(rgb_s');
	cov_t = cov(rgb_t');

	% decompose covariances
	[U_s,A_s,~] = svd(cov_s);
	[U_t,A_t,~] = svd(cov_t);

	rgbh_s = [rgb_s;ones(1,size(rgb_s,2))];

	% compute transforms
	% translations
	T_t = eye(4); T_t(1:3,4) = mean_t;
	T_s = eye(4); T_s(1:3,4) = -mean_s;
	% rotations
	R_t = blkdiag(U_t,1); R_s = blkdiag(inv(U_s),1);

	% scalings	
	% I added a 0.5 power to correct it.
	S_t = diag([diag(A_t).^(0.5);1]);
	S_s = diag([diag(A_s).^(-0.5);1]);

	rgbh_e = T_t * R_t * S_t * S_s * R_s * T_s * rgbh_s; % estimated RGBs
	rgbh_e = bsxfun(@rdivide, rgbh_e, rgbh_e(4,:));
	rgb_e = rgbh_e(1:3,:);

	dstImg = reshape(rgb_e',size(srcImg));

	end


## 2.3 Color Transfer using Principal Component Analysis ##

思路与2.2类似，主要是针对协方差矩阵进行了不同计算。

    function dstImg = ColorTransfer_by_PCA(srcImg,tgtImg)

	rgb_s = reshape(im2double(srcImg),[],3)';
	rgb_t = reshape(im2double(tgtImg),[],3)';

	% Covariance Matrixs
	CovMatrix_Ref=cov(rgb_s');
	CovMatrix_Tar=cov(rgb_t');

	% Eigenvalue Decomposition
	[U_Ref,Sigma_Ref]=eig(CovMatrix_Ref);
	[U_Tar,Sigma_Tar]=eig(CovMatrix_Tar);

	% Mean Color Vectors of ReferImg and TargetImg
	mean_s = mean(rgb_s,2);
	mean_t = mean(rgb_t,2);
	Diff=rgb_t-mean_t;

	Img=U_Ref*(U_Tar^(-1))*Diff+mean_s;
	
	dstImg = reshape(Img',size(source));

	end
