function [F] =husimi(phi,thet)
n=256;
cut_full=n+1;
A=su2cs(phi,thet,n);
%C=spin_gkp_grid_unequal(n+1,0.5,5);
%C=oat(n,1/sqrt(n));
C=tactgkp(cut_full,0.1,6);
F=abs(sum(conj(A).*C))^2;