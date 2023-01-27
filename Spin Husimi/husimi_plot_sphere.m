%clear all
%clc
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% this code plots the Husimi function on the sphere given that you have already a function to 
%% to find the Husimi function (works in general for any spherical functions like spin wigner function)
%% 

colorType = 0; %0 for regular, 1 for F=3,4 block colors
make_movie = 1;
saveImages = 1;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
theta = linspace(pi,0,60); %% I choose 100 samples for theta but should change according to the dats
phi = linspace(-pi,pi,80); %% again number of samples depend on the specific simulation
[Theta,Phi]=meshgrid(theta,phi);
 X = sin(Theta).* cos(Phi)
 Y = sin(Theta) .* sin(Phi)
 Z = cos(Theta)

%% the C is the data where we have the Husimi function
a=surf(X,Y,Z,C);
set(gca,'visible','off')
shading interp
axis on
daspect([2 2 2])
view([90 0]) %% this just determines the orientation of the sphere
colormap(jet(256));
