function [F] =spin_gkp_grid_unequal(cut_full,delt,T)
%% this state gives \mu=0 case
N=cut_full-1;
n=N;
[Jx,Jy,Jz]=make_fs(N/2);
A=zeros(cut_full,1);
A(1)=1;
F=zeros(cut_full,1);



for j=-T:1:T
    for l=-T:1:T
        a1=gamma(n+1);        
        b1=gamma(n/2+delt/2*(sqrt(pi*n*((2*j)^2+l^2)))+1);
        b2=gamma(n/2-delt/2*(sqrt(pi*n*((2*j)^2+l^2)))+1);
        aa=a1/(b1*b2);
        zz=expm(-2*1i*(sqrt(2*(pi))*j)*Jy/sqrt(n));
        zz=zz*expm(2*1i*( (sqrt(pi/2))*l )*Jx/sqrt(n));
        F=F+(aa*(zz*A(:,end)));
    end
end
F=real(F)/norm(F);
end
