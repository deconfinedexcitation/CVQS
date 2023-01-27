th = linspace(pi,0,60) ;
phi = linspace(-pi,pi,80) ;
C=zeros(length(phi), length(th));
for j=1:length(phi)
    for k=1:length(th)
        C(j,k)=husimi(phi(j),th(k));
    end
end