function [F] =oat(n,t)
[Jx,Jy,Jz]=make_fs(n/2);
A=su2cs(0,pi/2,n);
A=expm(-(1i)*t*(Jz*Jz))*A;
F=A/sqrt(norm(A)^2);
