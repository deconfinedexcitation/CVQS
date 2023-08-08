function [F] =oversqueezed_oat(cut_full,t)
N=cut_full-1;
n=N;
[Jx,Jy,Jz]=make_fs(N/2);
Jplus=Jx+((1i)*Jy);
Jminus=Jx-((1i)*Jy);

F=su2cs(0,pi/2,n);
F=expm(-(1i)*t*(Jz*Jz))*F;
F=F/norm(F);
