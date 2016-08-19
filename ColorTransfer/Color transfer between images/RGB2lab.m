function [l,alpha,beta]= RGB2lab(im)
% Function by Ruchir Srivastava to convert an RGB image im to l,alpha,beta representation
imr=im(:,:,1);
imb=im(:,:,3);
img=im(:,:,2);
a1=size(imr);
b1=[.3811 .5783 .0402;.1967 .7244 .0782;.0241 .1288 .8444];
c1=[1/sqrt(3) 0 0;0 1/sqrt(6) 0;0 0 1/sqrt(2)];
d1=[1 1 1;1 1 -2;1 -1 0];
for i=1:a1(1)
    for j=1:a1(2)
        x=b1*double([imr(i,j) img(i,j) imb(i,j)]');
        L(i,j)= log10 (x(1));
        M(i,j)= log10 (x(2));
        S(i,j)= log10 (x(3));
        y=c1*d1*[L(i,j) M(i,j) S(i,j)]';
        l(i,j)= y(1);
        alpha(i,j)= y(2);
        beta(i,j)= y(3);
    end
end
end
