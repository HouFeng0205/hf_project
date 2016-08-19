clear
clc
close all

srcImg = imread('Src.png');
exImg = imread('Exp.png');

dst = ColorTransfer_by_PCA(srcImg, exImg);

figure; 
subplot(1,3,1); imshow(srcImg); title('Original Image'); axis off
subplot(1,3,2); imshow(exImg); title('Target Palette'); axis off
subplot(1,3,3); imshow(dst); title('Result After Colour Transfer'); axis off