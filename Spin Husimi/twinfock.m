function [F] =twinfock(cut_full)
N=cut_full-1;
n=N;
[Jx,Jy,Jz]=make_fs(N/2);
Jplus=Jx+((1i)*Jy);
Jminus=Jx-((1i)*Jy);

F=zeros(cut_full,1);
F((n/2)+1)=1;
