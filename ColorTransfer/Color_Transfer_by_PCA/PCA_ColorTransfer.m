% Function to implement the Color Transfer using PCA
% Principal Component Analysis.
%
% HAN Zifa, 20/11/2013
% City University of Hong Kong
clear;
clc;
close all;

% input images
ReferImg=imread('Src.png');%
TargetImg=imread('Exp.png');
figure;imshow(ReferImg);title('Reference Img');
figure;imshow(TargetImg);title('Target Img');

% Images Attributes
M1=size(ReferImg,1); %row
N1=size(ReferImg,2); %col
M2=size(TargetImg,1); %row
N2=size(TargetImg,2); %col

Sam_Ref=zeros(3,M1*N1); 
Sam_Ref(1,:)=ReferImg(1:M1*N1); 
Sam_Ref(2,:)=ReferImg(M1*N1+1:2*M1*N1); 
Sam_Ref(3,:)=ReferImg(2*M1*N1+1:3*M1*N1); 

Sam_Tar=zeros(3,M2*N2); 
Sam_Tar(1,:)=TargetImg(1:M2*N2); 
Sam_Tar(2,:)=TargetImg(M2*N2+1:2*M2*N2); 
Sam_Tar(3,:)=TargetImg(2*M2*N2+1:3*M2*N2); 

% Covariance Matrixs
CovMatrix_Ref=cov(Sam_Ref');
CovMatrix_Tar=cov(Sam_Tar');

% Eigenvalue Decomposition
[U_Ref,Sigma_Ref]=eig(CovMatrix_Ref);
[U_Tar,Sigma_Tar]=eig(CovMatrix_Tar);

% Mean Color Vectors of ReferImg and TargetImg

Mean_Ref=sum(Sam_Ref,2)/size(Sam_Ref,2);
Mean_Tar=sum(Sam_Tar,2)/size(Sam_Tar,2);

Diff=Sam_Tar-repmat(Mean_Tar,1,M2*N2);
% Diff1=Sam_Tar-repmat(Mean_Tar,1,M2*N2);

Img=U_Ref*(U_Tar^(-1))*Diff+repmat(Mean_Ref,1,M2*N2);
ColorTransfer_Img=zeros(M2,N2,3);
ColorTransfer_Img(1:M2*N2)=Img(1,:);
ColorTransfer_Img(M2*N2+1:2*M2*N2)=Img(2,:);
ColorTransfer_Img(2*M2*N2+1:3*M2*N2)=Img(3,:);

figure;imshow(uint8(ColorTransfer_Img));title('ColorTransfer Img');
