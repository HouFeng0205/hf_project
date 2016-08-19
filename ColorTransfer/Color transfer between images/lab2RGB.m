%Function by Ruchir Srivastava. function to convert an image from the l,alpha,beta to RGB
function im = lab2RGB(l,alpha,beta)
a2=size(l);
b2=[4.4687   -3.5887    0.1196
   -1.2197    2.3831   -0.1626
    0.0585   -0.2611    1.2057];
c2=[1/sqrt(3) 0 0;0 1/sqrt(6) 0;0 0 1/sqrt(2)];
d2=[1 1 1;1 1 -1;1 -2 0];
for i=1:a2(1)
    for j=1:a2(2)
        y=d2*c2*[l(i,j) alpha(i,j) beta(i,j)]';
        L(i,j)= power(10,(y(1)));
        M(i,j)= power(10,(y(2)));
        S(i,j)= power(10,(y(3)));
        x=b2*([L(i,j) M(i,j) S(i,j)]');
        R(i,j)= x(1);
        G(i,j)= x(2);
        B(i,j)= x(3);
    end
end
im(:,:,1)=R;
im(:,:,2)=G;
im(:,:,3)=B;
end
