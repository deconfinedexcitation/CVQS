function [fx, fy, fz] = make_fs(f)
% f is the collective spin and the inputs and outputs are [fx, fy, fz] = make_fs(f)
% f=N/2
d = 2*f+1;
fx=zeros(d);
for m=1:d
   for n=1:d
      if(m+1==n)
          fx(d-m+1,d-n+1)=(1/2)*sqrt((d-m)*m); % F_-
          fx(d-n+1,d-m+1)=(1/2)*sqrt((d-m)*m); % F_+
      end;
      
   end;
   
end;

fy=zeros(d);
for m=1:d
   for n=1:d
      if(m+1==n)
          fy(d-m+1,d-n+1)=1i*(1/2)*sqrt((d-m)*m); % F_-
          fy(d-n+1,d-m+1)=-1i*(1/2)*sqrt((d-m)*m); % F_+
      end;
      
   end;
   
end;

fz=-diag([-f:f]);
clear m n d 
