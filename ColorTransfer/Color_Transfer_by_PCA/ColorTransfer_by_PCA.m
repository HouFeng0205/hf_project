function dstImg = ColorTransfer_by_PCA(srcImg,tgtImg)

M1=size(srcImg,1); %row
N1=size(srcImg,2); %col

M2=size(tgtImg,1); %row
N2=size(tgtImg,2); %col

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
mean_s = repmat(mean_s,1,M2*N2);
mean_t = mean(rgb_t,2);
mean_t = repmat(mean_t,1,M2*N2);

Diff=rgb_t-mean_t;

Img=U_Ref*(U_Tar^(-1))*Diff+mean_s;
	
dstImg = uint8(reshape(Img',size(tgtImg))*255);

end