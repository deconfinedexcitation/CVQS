function [F] =su2cs(phi,thet,n)
[Jx,Jy,Jz]=make_fs(n/2);
Jminus=Jx-((1i)*Jy);
A=zeros(n+1,1);
A(1)=1;
A=expm(exp((1i)*phi)*tan(thet/2)*Jminus)*A;
F=A/sqrt(norm(A)^2);

