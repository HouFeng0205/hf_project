%performing the task of transfering color between images
%Function by Ruchir Srivastava
% See E. Reinhard et al, Color Transfer Between Images, in IEEE Computer Graphics and
% Application, 21(5), 2001.
%Inputs:    src and tgt are image matrices for source and target images
function colormatch(src,tgt)
subplot(1,3,1);
imshow(src);title('Source');
subplot(1,3,2);
imshow(tgt);title('Target');
[elt,alphat,betat]=RGB2lab(tgt);
[els,alphas,betas]=RGB2lab(src);
elp=impose(els,elt);            %  to impose the color of source(src) on the target(tgt) image
alphap=impose(alphas,alphat);
betap=impose(betas,betat);
im=lab2RGB(elp,alphap,betap);
subplot(1,3,3);
imshow(uint8(im));title('Color transfered from source to target');
end