function [fx, fy, fz] = new_fs(f)
% f is the collective spin and the inputs and outputs are [fx, fy, fz] = make_fs(f)
%Representation j=f of SU(2)
d = f+1;
N=f
fplus=zeros(d);
for n=2:N+1
    fplus(n-1,n)=sqrt((N-(n-1)+1)*(n-1));
end;

fminus=fplus';

fx=(fplus+fminus)/2;

fy=(fplus-fminus)/(2*(1i));

fz=-diag([-N/2:N/2]);
clear n m d N
