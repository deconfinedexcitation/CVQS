function [F] =tactgkp(cut_full,delt,T)
%% this state gives \mu=0 case
N=cut_full-1;
n=N;
[Jx,Jy,Jz]=make_fs(N/2);
Jplus=Jx+((1i)*Jy);
Jminus=Jx-((1i)*Jy);
A=zeros(cut_full,1);
A(1)=1;
F=zeros(cut_full,1);

mu=0;
dd=1/(delt^2);
z=atanh((dd-1)/(dd+1));
lamb=delt;

uuu=expm((z/(2*n))*((Jplus*Jplus) - (Jminus*Jminus)))*A;

a1=gammaln(n+1); 
d1=gammaln((n/2)+(lamb*sqrt(pi*n)*(mu/2))+1);
d2=gammaln((n/2)-(lamb*sqrt(pi*n)*(mu/2))+1);
zz=exp(a1-d1-d2)*expm(-2*(1i)*sqrt(pi/2)*mu*Jy/sqrt(n))*uuu;

for j=1:T
    b1=gammaln((n/2)+(lamb*sqrt(pi*n)*(j+(mu/2)))+1);
    b2=gammaln((n/2)-(lamb*sqrt(pi*n)*(j+(mu/2)))+1);
    c1=gammaln((n/2)+(lamb*sqrt(pi*n)*(j-(mu/2)))+1);
    c2=gammaln((n/2)-(lamb*sqrt(pi*n)*(j-(mu/2)))+1);

    zz=zz +(exp(a1-b1-b2)*expm(-2*1i*((sqrt(2*pi)*j) + (mu*sqrt(pi/2)))*Jy/sqrt(n))*uuu);
    zz=zz +(exp(a1-c1-c2)*expm(2*(1i)*((sqrt(2*pi)*j) - (mu*sqrt(pi/2)))*Jy/sqrt(n))*uuu);
end
F=zz;
F=F/norm(F);
