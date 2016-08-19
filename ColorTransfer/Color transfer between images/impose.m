function tp= impose(src,tgt)
%  to impose the color of source(src) on the target(tgt) image
%  Written by Ruchir Srivastava
 td=std2(tgt);
 tm=mean2(tgt);
 sd=std2(src);
 sm=mean2(src);
 tp=(tgt-tm).*(sd/td)+sm;
 end
