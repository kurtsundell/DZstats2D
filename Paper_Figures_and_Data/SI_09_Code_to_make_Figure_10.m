% Supporting code for Sundell and Macdonald (submitted) Geology for 2D KDE
% similarity comparison of global age-Hf data, or any other 2D data 

clear all
close all
clc

[filename pathname] = uigetfile({'*'},'File Selector'); %load the supplemental file with zircon age eHfT data

if ispc == 1
	fullpathname = char(strcat(pathname, '\', filename));
end
if ismac == 1
	fullpathname = char(strcat(pathname, '/', filename));
end

% Range of 2D data
xmin = 485;
xmax = 635;
ymin = -45;
ymax = 20;
xstep = 1;

% kernel bandwidths
bandwidth_x = 5;
bandwidth_y = 1;

offset = 15;

% how many pixels for the images, has to be in powers of 2, ne need to go over go over 2^12, results lookthe same
gridspc = 2^9;

% Read in data, format is name header and two columns of info, for our example we use age + Hf, but any 2D data will work
[numbers text1, data] = xlsread(fullpathname);
numbers = num2cell(numbers);

% Filter out any data that are not pairs of numbers
for i = 1:size(numbers,1)
	for j = 1:size(numbers,2)
		if cellfun('isempty', numbers(i,j)) == 0
			if cellfun(@isnan, numbers(i,j)) == 1
				numbers(i,j) = {[]};
			end	
		end
	end
end

% pull the names from the headers
for i = 1:(size(data,2)+1)/2
	Name(i,1) = data(1,i*2-1);
end

data_tmp = numbers(1:end,:); %use temporary variable
N = size(data_tmp,2)/2; % figure out how many samples

% Filter out any data not in the range set above
for k = 1:N
	for i = 1:length(data_tmp(:,1))
		if cellfun('isempty', data_tmp(i,k*2-1)) == 0 && cellfun('isempty', data_tmp(i,k*2)) == 0 
			if cell2num(data_tmp(i,k*2-1)) >= xmin && cell2num(data_tmp(i,k*2-1)) <= xmax && ...
					cell2num(data_tmp(i,k*2)) >= ymin && cell2num(data_tmp(i,k*2)) <= ymax
				data1(i,k*2-1:k*2) = cell2num(data_tmp(i,k*2-1:k*2));
			end
		end
	end	
end

% set min/max ranges for kde2d function
MIN_XY=[xmin-offset,ymin];
MAX_XY=[xmax+offset,ymax];

% jet colormap that clips 0 values
cmap =[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0400000000000000,0.0800000000000000,0.120000000000000,0.160000000000000,0.200000000000000,0.240000000000000,0.280000000000000,0.320000000000000,0.360000000000000,0.400000000000000,0.440000000000000,0.480000000000000,0.520000000000000,0.560000000000000,0.600000000000000,0.640000000000000,0.680000000000000,0.720000000000000,0.760000000000000,0.800000000000000,0.840000000000000,0.880000000000000,0.920000000000000,0.960000000000000,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.960000000000000,0.920000000000000,0.880000000000000,0.840000000000000,0.800000000000000,0.760000000000000,0.720000000000000,0.680000000000000,0.640000000000000,0.600000000000000,0.560000000000000,0.520000000000000;1,0,0,0,0,0,0,0,0,0,0,0,0,0.0400000000000000,0.0800000000000000,0.120000000000000,0.160000000000000,0.200000000000000,0.240000000000000,0.280000000000000,0.320000000000000,0.360000000000000,0.400000000000000,0.440000000000000,0.480000000000000,0.520000000000000,0.560000000000000,0.600000000000000,0.640000000000000,0.680000000000000,0.720000000000000,0.760000000000000,0.800000000000000,0.840000000000000,0.880000000000000,0.920000000000000,0.960000000000000,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.960000000000000,0.920000000000000,0.880000000000000,0.840000000000000,0.800000000000000,0.760000000000000,0.720000000000000,0.680000000000000,0.640000000000000,0.600000000000000,0.560000000000000,0.520000000000000,0.480000000000000,0.440000000000000,0.400000000000000,0.360000000000000,0.320000000000000,0.280000000000000,0.240000000000000,0.200000000000000,0.160000000000000,0.120000000000000,0.0800000000000000,0.0400000000000000,0,0,0,0,0,0,0,0,0,0,0,0,0;1,0.560000000000000,0.600000000000000,0.640000000000000,0.680000000000000,0.720000000000000,0.760000000000000,0.800000000000000,0.840000000000000,0.880000000000000,0.920000000000000,0.960000000000000,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.960000000000000,0.920000000000000,0.880000000000000,0.840000000000000,0.800000000000000,0.760000000000000,0.720000000000000,0.680000000000000,0.640000000000000,0.600000000000000,0.560000000000000,0.520000000000000,0.480000000000000,0.440000000000000,0.400000000000000,0.360000000000000,0.320000000000000,0.280000000000000,0.240000000000000,0.200000000000000,0.160000000000000,0.120000000000000,0.0800000000000000,0.0400000000000000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]';

% make 2D kdes and 2D cdfs for samples, save as 3D matrix
for k = 1:N
	data2 = data1(:,k*2-1:k*2);
	data2 = data2(any(data2 ~= 0,2),:);
	samplesizes(k,1) = length(data2(:,1));
	[bandwidth1,density1(:,:,k),X1,Y1] = kde2d_set_kernel(data2, gridspc, MIN_XY, MAX_XY, bandwidth_x, bandwidth_y);
	density1(:,:,k) = density1(:,:,k)./sum(sum(density1(:,:,k)));	
	CDF1_Q1(:,:,k) = cumsum(cumsum(density1(:,:,k),1),2);    % take the CDF of x at y = max (y) Local CDF for Quadrant 1
	CDF1_Q2(:,:,k) = cumsum(cumsum(density1(:,:,k), 1,'reverse'), 2);            %Local CDF for Quadrant 2
	CDF1_Q3(:,:,k) = cumsum(cumsum(density1(:,:,k), 1,'reverse'), 2, 'reverse'); %Local CDF for Quadrant 3
	CDF1_Q4(:,:,k) = cumsum(cumsum(density1(:,:,k), 1), 2, 'reverse');           %Loval CDF for Quadrant 4
	clear data2
	%waitbar(k/N, f, 'Calculating densities! Please wait...');
end
%2D quantitative comparison
count = 1;
for j = 1:N
	for k = 1:N
		name_comp(count,1) = strcat(Name(j,1), {' vs '}, Name(k,1));
		d1 = reshape(density1(:,:,j),size(density1,1)*size(density1,2),1);
		d2 = reshape(density1(:,:,k),size(density1,1)*size(density1,2),1);
		R2D(j,k) = ((sum((d1 - mean(d1)).*(d2 - mean(d2))))/(sqrt((sum((d1 - mean(d1)).*(d1 - mean(d1))))*(sum((d2 - mean(d2)).*(d2 - mean(d2)))))))^2;
		for m = 1:size(density1,1)
			for n = 1:size(density1,2)
				Maps_L(m,n,count) = (abs(density1(m,n,j) - density1(m,n,k))/2); %Likeness map
				Maps_S(m,n,count) = sqrt(density1(m,n,j).*density1(m,n,k)); % Similarity map
			end
		end
		L2D(j,k) = 1 - sum(Maps_L(:,:,count),'All');
		if j == k
			S2D(j,k) = 1;
		else
			S2D(j,k) = sum(Maps_S(:,:,count),'All');
		end
		count = count+1;
		D2Dtmp(j,k,1) = max(max(abs(CDF1_Q1(:,:,j) - CDF1_Q1(:,:,k)),[],1)); %Maximum absolute difference for Quadrant 1
		D2Dtmp(j,k,2) = max(max(abs(CDF1_Q2(:,:,j) - CDF1_Q2(:,:,k)),[],1)); %Maximum absolute difference for Quadrant 2
		D2Dtmp(j,k,3) = max(max(abs(CDF1_Q3(:,:,j) - CDF1_Q3(:,:,k)),[],1)); %Maximum absolute difference for Quadrant 3
		D2Dtmp(j,k,4) = max(max(abs(CDF1_Q4(:,:,j) - CDF1_Q4(:,:,k)),[],1)); %Maximum absolute difference for Quadrant 4
		D2D = max(D2Dtmp, [], 3);		
		V2Dtmp(j,k,1) = max( max(CDF1_Q1(:,:,j) - CDF1_Q1(:,:,k),[],1) + max(CDF1_Q1(:,:,k) - CDF1_Q1(:,:,j),[],1) );
		V2Dtmp(j,k,2) = max( max(CDF1_Q2(:,:,j) - CDF1_Q2(:,:,k),[],1) + max(CDF1_Q2(:,:,k) - CDF1_Q2(:,:,j),[],1) );
		V2Dtmp(j,k,3) = max( max(CDF1_Q3(:,:,j) - CDF1_Q3(:,:,k),[],1) + max(CDF1_Q3(:,:,k) - CDF1_Q3(:,:,j),[],1) );
		V2Dtmp(j,k,4) = max( max(CDF1_Q4(:,:,j) - CDF1_Q4(:,:,k),[],1) + max(CDF1_Q4(:,:,k) - CDF1_Q4(:,:,j),[],1) );		
		V2Dtmp(j,k,5) = max( max(CDF1_Q1(:,:,j) - CDF1_Q1(:,:,k),[],2) + max(CDF1_Q1(:,:,k) - CDF1_Q1(:,:,j),[],2) );
		V2Dtmp(j,k,6) = max( max(CDF1_Q2(:,:,j) - CDF1_Q2(:,:,k),[],2) + max(CDF1_Q2(:,:,k) - CDF1_Q2(:,:,j),[],2) );
		V2Dtmp(j,k,7) = max( max(CDF1_Q3(:,:,j) - CDF1_Q3(:,:,k),[],2) + max(CDF1_Q3(:,:,k) - CDF1_Q3(:,:,j),[],2) );
		V2Dtmp(j,k,8) = max( max(CDF1_Q4(:,:,j) - CDF1_Q4(:,:,k),[],2) + max(CDF1_Q4(:,:,k) - CDF1_Q4(:,:,j),[],2) );
		V2D = max(V2Dtmp,[],3);
	end
end

% plot and save to folder on Desktop named 'figs_Andes'
for i = 1:N
	F1 = figure;
	hold on
	surf(X1,Y1,density1(:,:,i));
	colormap(cmap)
	shading interp
	view(2)
	title(Name(i))
	axis([xmin-offset xmax+offset ymin ymax])
	%saveas(F1,char(strcat('/Users/kurtsundell/Desktop/figs/', Name(i,1),{'_MAP.eps'})),'epsc')
	
	max_density1S = max(max(density1(:,:,i)));
	max_density_confS = max_density1S*.01; %clip lowest colormap value to avoid excessive use of color
	F2 = figure;
	contour3(X1,Y1,density1(:,:,i),[max_density_confS max_density_confS],'k', 'LineWidth', 3);
	grid off
	view(2)
	axis([xmin-offset xmax+offset ymin ymax])
	%print(F2,'-depsc','-painters',char(strcat('/Users/kurtsundell/Desktop/figs/', Name(i,1),{'_CONT.eps'})));
	%epsclean(char(strcat('/Users/kurtsundell/Desktop/figs/', Name(i,1),{'_CONT.eps'}))); % save simplified contours
end

F3 = figure;
hold on
surf(X1,Y1,Maps_L(:,:,2));
colormap(cmap)
shading interp
view(2)
title('Mismatch')
axis([xmin-offset xmax+offset ymin ymax])
%saveas(F3,char(strcat('/Users/kurtsundell/Desktop/figs/', 'Mismatch',{'_MAP.eps'})),'epsc')

max_density1S = max(max(Maps_L(:,:,2)));
max_density_confS = max_density1S*.01; %clip lowest colormap value to avoid excessive use of color
F4 = figure;
contour3(X1,Y1,Maps_L(:,:,2),[max_density_confS max_density_confS],'k', 'LineWidth', 3);
grid off
view(2)
axis([xmin-offset xmax+offset ymin ymax])
%print(F4,'-depsc','-painters',char(strcat('/Users/kurtsundell/Desktop/figs/', 'Mismatch',{'_CONT.eps'})));
%epsclean(char(strcat('/Users/kurtsundell/Desktop/figs/', 'Mismatch',{'_CONT.eps'}))); % save simplified contours

F5 = figure;
hold on
surf(X1,Y1,Maps_S(:,:,2));
colormap(cmap)
shading interp
view(2)
title('Similarity')
axis([xmin-offset xmax+offset ymin ymax])
%saveas(F5,char(strcat('/Users/kurtsundell/Desktop/figs/', 'Similarity',{'_MAP.eps'})),'epsc')

max_density1S = max(max(Maps_S(:,:,2)));
max_density_confS = max_density1S*.01; %clip lowest colormap value to avoid excessive use of color
F6 = figure;
contour3(X1,Y1,Maps_S(:,:,2),[max_density_confS max_density_confS],'k', 'LineWidth', 3);
grid off
view(2)
axis([xmin-offset xmax+offset ymin ymax])
%print(F6,'-depsc','-painters',char(strcat('/Users/kurtsundell/Desktop/figs/', 'Similarity',{'_CONT.eps'})));
%epsclean(char(strcat('/Users/kurtsundell/Desktop/figs/', 'Similarity',{'_CONT.eps'}))); % save simplified contours






function [a] = kde(m, s, xmin, xmax, xint)
x = xmin:xint:xmax;
n = length(m);
f = zeros(n,length(x));
for i = 1:n;
	f(i,:) = (1./ (s(i)*sqrt(2*pi)) .* exp (  (-((x-m(i)).^2)) ./ (2*((s(i)).^2))  ).*xint); % Gaussian
end
a = (sum(f))/n; %sum and normalize
end

%2D KDE algorithm sourced from
%https://github.com/rctorres/Matlab/blob/master/kde2d.m or 
%https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/17204/versions/5/previews/kde2d.m/index.html
%Modified by Sundell -- added bandwidth_x and bandwidth_y to the original code for set bandwidths (i.e., no Botev et al. (2010) algorithm)
function [bandwidth,density,X,Y]=kde2d_set_kernel(data,n,MIN_XY,MAX_XY,bandwidth_x,bandwidth_y) 
% bivariate kernel density estimator
% with diagonal bandwidth matrix.
% The kernel is assumed to be Gaussian.
% The two bandwidth parameters are
% chosen optimally without ever
% using/assuming a parametric model for the data or any "rules of thumb".
% Unlike many other procedures, this one
% is immune to accuracy failures in the estimation of
% multimodal densities with widely separated modes (see examples).
% INPUTS: data - an N by 2 array with continuous data
%            n - size of the n by n grid over which the density is computed
%                n has to be a power of 2, otherwise n=2^ceil(log2(n));
%                the default value is 2^8;
% MIN_XY,MAX_XY- limits of the bounding box over which the density is computed;
%                the format is:
%                MIN_XY=[lower_Xlim,lower_Ylim]
%                MAX_XY=[upper_Xlim,upper_Ylim].
%                The dafault limits are computed as:
%                MAX=max(data,[],1); MIN=min(data,[],1); Range=MAX-MIN;
%                MAX_XY=MAX+Range/4; MIN_XY=MIN-Range/4;
% OUTPUT: bandwidth - a row vector with the two optimal
%                     bandwidths for a bivaroate Gaussian kernel;
%                     the format is:
%                     bandwidth=[bandwidth_X, bandwidth_Y];
%          density  - an n by n matrix containing the density values over the n by n grid;
%                     density is not computed unless the function is asked for such an output;
%              X,Y  - the meshgrid over which the variable "density" has been computed;
%                     the intended usage is as follows:
%                     surf(X,Y,density)
% Example (simple Gaussian mixture)
% clear all
%   % generate a Gaussian mixture with distant modes
%   data=[randn(500,2);
%       randn(500,1)+3.5, randn(500,1);];
%   % call the routine
%     [bandwidth,density,X,Y]=kde2d(data);
%   % plot the data and the density estimate
%     contour3(X,Y,density,50), hold on
%     plot(data(:,1),data(:,2),'r.','MarkerSize',5)
%
% Example (Gaussian mixture with distant modes):
%
% clear all
%  % generate a Gaussian mixture with distant modes
%  data=[randn(100,1), randn(100,1)/4;
%      randn(100,1)+18, randn(100,1);
%      randn(100,1)+15, randn(100,1)/2-18;];
%  % call the routine
%    [bandwidth,density,X,Y]=kde2d(data);
%  % plot the data and the density estimate
%  surf(X,Y,density,'LineStyle','none'), view([0,60])
%  colormap hot, hold on, alpha(.8)
%  set(gca, 'color', 'blue');
%  plot(data(:,1),data(:,2),'w.','MarkerSize',5)
%
% Example (Sinusoidal density):
%
% clear all
%   X=rand(1000,1); Y=sin(X*10*pi)+randn(size(X))/3; data=[X,Y];
%  % apply routine
%  [bandwidth,density,X,Y]=kde2d(data);
%  % plot the data and the density estimate
%  surf(X,Y,density,'LineStyle','none'), view([0,70])
%  colormap hot, hold on, alpha(.8)
%  set(gca, 'color', 'blue');
%  plot(data(:,1),data(:,2),'w.','MarkerSize',5)
%
%  Reference:
% Kernel density estimation via diffusion
% Z. I. Botev, J. F. Grotowski, and D. P. Kroese (2010)
% Annals of Statistics, Volume 38, Number 5, pages 2916-2957.

global N A2 I
if nargin<2
    n=2^8;
end
n=2^ceil(log2(n)); % round up n to the next power of 2;
N=size(data,1);
if nargin<3
    MAX=max(data,[],1); MIN=min(data,[],1); Range=MAX-MIN;
    MAX_XY=MAX+Range/2; MIN_XY=MIN-Range/2;
end
scaling=MAX_XY-MIN_XY;
if N<=size(data,2)
    error('data has to be an N by 2 array where each row represents a two dimensional observation')
end
transformed_data=(data-repmat(MIN_XY,N,1))./repmat(scaling,N,1);
%bin the data uniformly using regular grid;
initial_data=ndhist(transformed_data,n);

% discrete cosine transform of initial data
a= dct2d(initial_data);

% now compute the optimal bandwidth^2
  I=(0:n-1).^2; A2=a.^2;
 t_star=root(@(t)(t-evolve(t)),N);
p_02=func([0,2],t_star);p_20=func([2,0],t_star); p_11=func([1,1],t_star);

t_x=(p_20^(3/4)/(4*pi*N*p_02^(3/4)*(p_11+sqrt(p_20*p_02))))^(1/3);
t_y=(p_02^(3/4)/(4*pi*N*p_20^(3/4)*(p_11+sqrt(p_20*p_02))))^(1/3);

%bandwidth_opt = sqrt([t_x,t_y]).*scaling;

% Sundell modified this bit for set kernels
bandwidth = [bandwidth_x, bandwidth_y];
t_x = (bandwidth_x(1,1)/scaling(1,1))^2;
t_y = (bandwidth_y(1,1)/scaling(1,2))^2;

% smooth the discrete cosine transform of initial data using t_star
a_t=exp(-(0:n-1)'.^2*pi^2*t_x/2)*exp(-(0:n-1).^2*pi^2*t_y/2).*a; 

% now apply the inverse discrete cosine transform
if nargout>1
    density=idct2d(a_t)*(numel(a_t)/prod(scaling));
	density(density<0)=eps; % remove any negative density values
    [X,Y]=meshgrid(MIN_XY(1):scaling(1)/(n-1):MAX_XY(1),MIN_XY(2):scaling(2)/(n-1):MAX_XY(2));
end

end
%#######################################
function  [out,time]=evolve(t)
global N
Sum_func = func([0,2],t) + func([2,0],t) + 2*func([1,1],t);
time=(2*pi*N*Sum_func)^(-1/3);
out=(t-time)/time;
end
%#######################################
function out=func(s,t)
global N
if sum(s)<=4
    Sum_func=func([s(1)+1,s(2)],t)+func([s(1),s(2)+1],t); const=(1+1/2^(sum(s)+1))/3;
    time=(-2*const*K(s(1))*K(s(2))/N/Sum_func)^(1/(2+sum(s)));
    out=psi(s,time);
else
    out=psi(s,t);
end

end
%#######################################
function out=psi(s,Time)
global I A2
% s is a vector
w=exp(-I*pi^2*Time).*[1,.5*ones(1,length(I)-1)];
wx=w.*(I.^s(1));
wy=w.*(I.^s(2));
out=(-1)^sum(s)*(wy*A2*wx')*pi^(2*sum(s));
end
%#######################################
function out=K(s)
out=(-1)^s*prod((1:2:2*s-1))/sqrt(2*pi);
end
%#######################################
function data=dct2d(data)
% computes the 2 dimensional discrete cosine transform of data
% data is an nd cube
[nrows,ncols]= size(data);
if nrows~=ncols
    error('data is not a square array!')
end
% Compute weights to multiply DFT coefficients
w = [1;2*(exp(-i*(1:nrows-1)*pi/(2*nrows))).'];
weight=w(:,ones(1,ncols));
data=dct1d(dct1d(data)')';
    function transform1d=dct1d(x)

        % Re-order the elements of the columns of x
        x = [ x(1:2:end,:); x(end:-2:2,:) ];

        % Multiply FFT by weights:
        transform1d = real(weight.* fft(x));
    end
end
%#######################################
function data = idct2d(data)
% computes the 2 dimensional inverse discrete cosine transform
[nrows,ncols]=size(data);
% Compute wieghts
w = exp(i*(0:nrows-1)*pi/(2*nrows)).';
weights=w(:,ones(1,ncols));
data=idct1d(idct1d(data)');
    function out=idct1d(x)
        y = real(ifft(weights.*x));
        out = zeros(nrows,ncols);
        out(1:2:nrows,:) = y(1:nrows/2,:);
        out(2:2:nrows,:) = y(nrows:-1:nrows/2+1,:);
    end
end
%#######################################
function binned_data=ndhist(data,M)
% this function computes the histogram
% of an n-dimensional data set;
% 'data' is nrows by n columns
% M is the number of bins used in each dimension
% so that 'binned_data' is a hypercube with
% size length equal to M;
[nrows,ncols]=size(data);
bins=zeros(nrows,ncols);
for i=1:ncols
    [dum,bins(:,i)] = histc(data(:,i),[0:1/M:1],1);
    bins(:,i) = min(bins(:,i),M);
end
% Combine the  vectors of 1D bin counts into a grid of nD bin
% counts.
binned_data = accumarray(bins(all(bins>0,2),:),1/nrows,M(ones(1,ncols)));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function t=root(f,N)
% try to find smallest root whenever there is more than one
N=50*(N<=50)+1050*(N>=1050)+N*((N<1050)&(N>50));
tol=10^-12+0.01*(N-50)/1000;
flag=0;
while flag==0
    try
        t=fzero(f,[0,tol]);
        flag=1;
    catch
        tol=min(tol*2,.1); % double search interval
    end
    if tol==.1 % if all else fails
        t=fminbnd(@(x)abs(f(x)),0,.1); flag=1;
    end
end
end

%cell2num sourced from https://www.mathworks.com/matlabcentral/fileexchange/15306-cell2num
function [outputmat]=cell2num(inputcell)
% Function to convert an all numeric cell array to a double precision array
% ********************************************
% Usage: outputmatrix=cell2num(inputcellarray)
% ********************************************
% Output matrix will have the same dimensions as the input cell array
% Non-numeric cell contest will become NaN outputs in outputmat
% This function only works for 1-2 dimensional cell arrays

if ~iscell(inputcell), error('Input cell array is not.'); end

outputmat=zeros(size(inputcell));

for c=1:size(inputcell,2)
  for r=1:size(inputcell,1)
    if isnumeric(inputcell{r,c})
      outputmat(r,c)=inputcell{r,c};
    else
      outputmat(r,c)=NaN;
    end
  end  
end

end

function epsclean( file, varargin )
% EPSCLEAN Cleans up a MATLAB exported .eps file.
%
%   EPSCLEAN(F,...) cleans the .eps file F without removing box elements and optional parameters.
%   EPSCLEAN(F,O,...) cleans the .eps file F, writes the result to file O and optional parameters.
%   EPSCLEAN(F,O,R,G) (deprecated) cleans the .eps file F, writes the result to file O and optionally removes box
%                     elements if R = true. Optionally it groups elements 'softly' if G = true.
%
%   Optional parameters (key/value pairs) - see examples below
%   - outFile      ... Defines the output file for the result. Default is overwriting the input file.
%   - groupSoft    ... Groups elements only if they occur sequentially. Can help with Z-order problems. Defaults to false.
%   - combineAreas ... Combines filled polygons to larger ones. Can help with artifacts. Defaults to false.
%   - removeBoxes  ... Removes box (rectangle) elements. Defaults to false.
%   - closeGaps    ... For every filled polygon, also draw a fine polyline to close potential gaps between adjacent polygon areas. Defaults to false.
%   - gapWidth     ... The width of polylines to cover gaps. Defaults to 0.01.
%
%   When exporting a figure with Matlab's 'saveas' function to vector graphics multiple things might occur:
%   - Paths are split up into multiple segments and white lines are created on patch objects
%     see https://de.mathworks.com/matlabcentral/answers/290313-why-is-vector-graphics-chopped-into-pieces
%   - There are unnecessary box elements surrounding the paths
%   - Lines which actually should be continuous are split up in small line segments
%
%   Especially the fragmentation is creating highly unusable vector graphics for further post-processing.
%   This function fixes already exported figures in PostScript file format by grouping paths together according to their
%   properties (line width, line color, transformation matrix, ...). Small line segments which logically should belong
%   together are replaced by one continous line.
%   It also removes paths with 're' (rectangle) elements when supplying the parameter 'removeBoxes' with true.
%   In case the 'groupSoft' parameter is true it does not group elements according to their properties over the whole
%   document. It will rather group them only if the same elements occur sequentially, but not if they are interrupted by
%   elements with different properties. This will result in more fragmentation, but the Z-order will be kept intact. Use
%   this (set to true) if you have trouble with the Z-order.
%   If the 'combineAreas' parameter is true it combines filled polygons with the same properties to larger polygons of
%   the same type. It reduces clutter and white-line artifacts. The downside is that it's about 10 times slower.
%
%   Example 1
%   ---------
%       z = peaks;
%       contourf(z);
%       print(gcf,'-depsc','-painters','out.eps');
%       epsclean('out.eps'); % cleans and overwrites the input file
%       epsclean('out.eps','clean.eps'); % leaves the input file intact
%       epsclean('out.eps','clean.eps','combineAreas',true); % result in 'clean.eps', combines polygon areas
%       epsclean('out.eps','groupSoft',true,'combineAreas',true); % overwrites file, combines polygons, Z-order preserved
%
%   Example 2
%   ---------
%       [X,Y,Z] = peaks(100);
%       [~,ch] = contourf(X,Y,Z);
%       ch.LineStyle = 'none';
%       ch.LevelStep = ch.LevelStep/10;
%       colormap('hot')
%       saveas(gcf, 'out.eps', 'epsc');
%       epsclean('out.eps');
%
%   Notes
%   -----
%   - A block is a starting with GS (gsave) and ends with GR (grestore)
%   - Only text after %%EndPageSetup is analyzed
%   - Removing boxes will also remove the clipping area (if any)
%   - Tested on Windows with Matlab R2016b
%
%   Changes
%   -------
%   2017-04-03 (YYYY-MM-DD)
%   - Line segments with the same properties are converted to one continous polyline
%      o As a side effect this will cause multiple equal lines on top of each other to merge
%   - The Z-order of elements can be preserved by using 'groupSoft = true'
%      o See https://github.com/Conclusio/matlab-epsclean/issues/6
%      o This will cause additional fragmentation which might or might not be what you want
%   2017-04-18 (YYYY-MM-DD)
%   - Major performance increase for creating the adjacency matrix (for creating continous polylines)
%   - A lot of other performance enhancements
%   2017-05-28 (YYYY-MM-DD)
%   - Added the possibility to merge adjacent polygons to avoid artifacts
%     o See https://github.com/Conclusio/matlab-epsclean/issues/9
%   - Changed argument style
%   2018-04-12 (YYYY-MM-DD)
%   - Added parameter 'closeGaps' to hide lines between filled areas
%     o See https://github.com/Conclusio/matlab-epsclean/issues/9
%   - Added parameter 'gapWidth' to control the line width
%
%   ------------------------------------------------------------------------------------------
%   Copyright 2017,2018, Stefan Spelitz, Vienna University of Technology (TU Wien).
%   This code is distributed under the terms of the GNU Lesser General Public License (LGPL).
%
%   This program is free software: you can redistribute it and/or modify
%   it under the terms of the GNU Lesser General Public License as published by
%   the Free Software Foundation, either version 3 of the License, or
%   (at your option) any later version.
% 
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU Lesser General Public License for more details.
% 
%   You should have received a copy of the GNU Lesser General Public License
%   along with this program.  If not, see <http://www.gnu.org/licenses/>.

% default values:
removeBoxes = false;
groupSoft = false;
combineAreas = false;
closeGaps = false;
gapWidth = 0.01;
outfile = file;

fromIndex = 1;
% check for old argument style (backward compatibility)
if nargin >= 2 && ischar(varargin{1}) && ~strcmpi(varargin{1},'removeBoxes') && ~strcmpi(varargin{1},'groupSoft') && ~strcmpi(varargin{1},'combineAreas') && ~strcmpi(varargin{1},'closeGaps') && ~strcmpi(varargin{1},'gapWidth')
    fromIndex = 2;
    outfile = varargin{1};
    if nargin >= 3
        if islogical(varargin{2})
            fromIndex = 3;
            removeBoxes = varargin{2};
            if nargin >= 4 && islogical(varargin{3})
                fromIndex = 4;
                groupSoft = varargin{3};
            end
        end
    end
end

p = inputParser;
p.CaseSensitive = false;
p.KeepUnmatched = false;

addParameter(p,'outFile',outfile,@ischar);
addParameter(p,'removeBoxes',removeBoxes,@islogical);
addParameter(p,'groupSoft',groupSoft,@islogical);
addParameter(p,'combineAreas',combineAreas,@islogical);
addParameter(p,'closeGaps',closeGaps,@islogical);
addParameter(p,'gapWidth',gapWidth,@isfloat);

parse(p,varargin{fromIndex:end});
outfile = p.Results.outFile;
removeBoxes = p.Results.removeBoxes;
groupSoft = p.Results.groupSoft;
combineAreas = p.Results.combineAreas;
closeGaps = p.Results.closeGaps;
gapWidth = p.Results.gapWidth;

keepInput = true;
if strcmp(file, outfile)
    outfile = [file '_out']; % tmp file
    keepInput = false;
end

fid1 = fopen(file,'r');
fid2 = fopen(outfile,'W');

previousBlockPrefix = [];
operation = -1; % -1 .. wait for 'EndPageSetup', 0 .. wait for blocks, 1 .. create id, 2 .. analyze block content, 3 .. analyzed
insideAxg = false;
blockGood = true;
hasLineCap = false;
isDashMode = false;
blockList = [];

nested = 0;
lastMLine = [];
lastLLine = [];
blockMap = containers.Map(); % key=blockPrefix -> MAP with connection information and content for blocks

% current block (cb) data:
cbPrefix = '';
cbContentLines = -ones(1,100);
cbContentLinesFull = -ones(1,100);
cbContentLinesIdx = 1;
cbContentLinesFullIdx = 1;
cbConn = {};
cbIsFill = false;

% load whole file into memory:
fileContent = textscan(fid1,'%s','delimiter','\n','whitespace','');
fileContent = fileContent{1}';
lineCount = length(fileContent);
lineIdx = 0;

while lineIdx < lineCount
    lineIdx = lineIdx + 1;
    thisLine = cell2mat(fileContent(lineIdx));
    
    % normal read until '%%EndPageSetup'
    if operation == -1
        if closeGaps && startsWith(thisLine,'/f/fill')
            fileContent(lineIdx) = { sprintf('/f{GS %.5f setlinewidth S GR fill}bd', gapWidth) };
        elseif equalsWith(thisLine, '%%EndPageSetup')
            operation = 0;
            fprintf(fid2, '%s\n', strjoin(fileContent(1:lineIdx),'\n')); % dump prolog
        end
        continue;
    end
    
    if operation == 3 % block was analyzed
        if blockGood
            if groupSoft && ~strcmp(cbPrefix, previousBlockPrefix)
                % SOFT GROUPING. different block -> dump all existent ones except the current one

                currentBlock = [];
                if blockMap.isKey(cbPrefix)
                    currentBlock = blockMap(cbPrefix);
                    blockMap.remove(cbPrefix);
                end
                
                writeBlocks(blockList, blockMap, fid2, fileContent);
                
                blockList = [];
                blockMap = containers.Map();
                if ~isempty(currentBlock)
                    blockMap(cbPrefix) = currentBlock;
                end
            end

            [cbNewBlock,oldConn,oldConnFill] = getBlockData(blockMap,cbPrefix);
            removeLastContentLine = false;
            if cbIsFill
                if combineAreas
                    oldConnFill = [oldConnFill cbConn]; %#ok<AGROW>
                else
                    removeLastContentLine = true;
                end
            else
                oldConn = [oldConn cbConn]; %#ok<AGROW>
            end
            setBlockData(blockMap,cbPrefix,cbContentLines(1:cbContentLinesIdx-1),oldConn,oldConnFill,removeLastContentLine);
            if cbNewBlock
                % new block
                block = struct('prefix', cbPrefix);
                blockList = [blockList block]; %#ok<AGROW>
            end
        end
        operation = 0;
        previousBlockPrefix = cbPrefix;
        cbPrefix = '';
    end


    if operation == 0 % waiting for blocks
        if equalsWith(thisLine,'GS')
            % start of a block
            operation = 1;
            hasLineCap = false;
            isDashMode = false;
            nested = 0;
        elseif equalsWith(thisLine,'%%Trailer')
            % end of figures -> dump all blocks
            writeBlocks(blockList, blockMap, fid2, fileContent);
            fprintf(fid2, '%s\n', thisLine);
        elseif equalsWith(thisLine,'GR')
            % unexpected GR before a corresponding GS -> ignore
        else
            % not inside a block and not the start of a block -> just take it
            fprintf(fid2, '%s\n', thisLine);
        end
    elseif operation == 1 % inside GS/GR block
        % build prefix
        if startsWith(thisLine,'%AXGBegin')
            % this could be the beginning of a raw bitmap data block -> just take it
            insideAxg = true;
            cbPrefix = sprintf('%s%s\n',cbPrefix, thisLine);
        elseif startsWith(thisLine,'%AXGEnd')
            insideAxg = false;
            cbPrefix = sprintf('%s%s\n',cbPrefix, thisLine);
        elseif insideAxg
            cbPrefix = sprintf('%s%s\n',cbPrefix, thisLine);
        elseif equalsWith(thisLine,'N')
            % begin analyzing
            operation = 2;
            blockGood = true;
            cbContentLinesIdx = 1;
            cbContentLinesFullIdx = 1;
            lastMLine = [];
            cbConn = {};
            cbIsFill = false;
        elseif equalsWith(thisLine,'GS')
            nested = nested + 1;
            cbPrefix = sprintf('%s%s\n',cbPrefix, thisLine);
        elseif equalsWith(thisLine,'GR')
            nested = nested - 1;
            if nested >= 0
                cbPrefix = sprintf('%s%s\n',cbPrefix, thisLine);
            else
                % end of block without a 'N' = newpath command
                % we don't know what it is, but we take it as a whole
                operation = 3;
                blockGood = true;
                cbContentLinesIdx = 1;
                cbContentLinesFullIdx = 1;
                cbConn = {};
                cbIsFill = false;
            end
        elseif endsWith(thisLine,'setdash')
            isDashMode = true;
            cbPrefix = sprintf('%s%s\n',cbPrefix, thisLine);
        elseif endsWith(thisLine,'setlinecap')
            hasLineCap = true;
            cbPrefix = sprintf('%s%s\n',cbPrefix, thisLine);
        elseif endsWith(thisLine,'LJ')
            if hasLineCap
                cbPrefix = sprintf('%s%s\n',cbPrefix, thisLine);
            elseif ~isDashMode
                % add '1 linecap' if no linecap is specified
                cbPrefix = sprintf('%s%s\n%s\n',cbPrefix,'1 setlinecap',thisLine);
            end
        else
            cbPrefix = sprintf('%s%s\n',cbPrefix, thisLine);
        end
    elseif operation == 2 % analyze block content
        if startsWith(thisLine,'%AXGBegin')
            % this could be the beginning of a raw bitmap data block -> just take it
            insideAxg = true;
            [cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx] = addContent(cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx,lineIdx,true);
        elseif startsWith(thisLine,'%AXGEnd')
            insideAxg = false;
            [cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx] = addContent(cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx,lineIdx,true);
        elseif insideAxg
            [cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx] = addContent(cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx,lineIdx,true);
        elseif endsWith(thisLine,'re')
            if removeBoxes
                blockGood = false;
            else
                [cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx] = addContent(cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx,lineIdx,true);
            end
        elseif equalsWith(thisLine,'clip')
            cbPrefix = sprintf('%sN\n%s\n%s\n', cbPrefix, strjoin(fileContent(cbContentLinesFull(1:cbContentLinesFullIdx-1))), thisLine);
            cbContentLinesIdx = 1;
            cbContentLinesFullIdx = 1;
            cbConn = {};
            cbIsFill = false;
        elseif endsWith(thisLine,'M')
            lastMLine = thisLine;
            lineIdx = lineIdx + 1;
            nextline = cell2mat(fileContent(lineIdx)); % ASSUMPTION: there is an L directly after an M
            lastLLine = nextline;
            
            moveId = thisLine(1:end-1);
            lineId = nextline(1:end-1);
            
            [cbConn] = addConnection(moveId,lineId,cbConn);
            [cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx] = addContent(cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx,lineIdx-1,false);
            [cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx] = addContent(cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx,lineIdx,false);
        elseif equalsWith(thisLine,'cp')
            moveId = lastLLine(1:end-1);
            lineId = lastMLine(1:end-1);
            lastLLine = lastMLine;

            [cbConn] = addConnection(moveId,lineId,cbConn);
            [cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx] = addContent(cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx,lineIdx,false);
        elseif endsWith(thisLine,'L')
            moveId = lastLLine(1:end-1);
            lineId = thisLine(1:end-1);
            lastLLine = thisLine;

            [cbConn] = addConnection(moveId,lineId,cbConn);
            [cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx] = addContent(cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx,lineIdx,false);
        elseif equalsWith(thisLine,'S')
            % ignore stroke command
        elseif equalsWith(thisLine,'f')
            % special handling for filled areas
            cbIsFill = true;
            if combineAreas
                lastLine = cell2mat(fileContent(lineIdx-1));
                if ~equalsWith(lastLine, 'cp')
                    [cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx] = addContent(cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx,lineIdx,true);
                end
            else
                [~,cbContentLinesFull,~,cbContentLinesFullIdx] = addContent(cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx,lineIdx,false);
                cbContentLines = cbContentLinesFull;
                cbContentLinesIdx = cbContentLinesFullIdx;
                % remove all connections:
                cbConn = {};
            end
        elseif equalsWith(thisLine,'GS')
            nested = nested + 1;
            [cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx] = addContent(cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx,lineIdx,true);
        elseif equalsWith(thisLine,'GR')
            % end of block content
            nested = nested - 1;
            if nested >= 0
                [cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx] = addContent(cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx,lineIdx,true);
            else
                operation = 3; % end of block content
            end
        else
            [cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx] = addContent(cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx,lineIdx,true);
        end
    end
        
end %while

fclose(fid1);
fclose(fid2);

if ~keepInput
    delete(file);
    movefile(outfile, file, 'f');
end

end

function r = startsWith(string1, pattern)
    l = length(pattern);
    if length(string1) < l
        r = false;
    else
        r = strcmp(string1(1:l),pattern);
    end
end

function r = endsWith(string1, pattern)
    l = length(pattern);
    if length(string1) < l
        r = false;
    else
        r = strcmp(string1(end-l+1:end),pattern);
    end
end

function r = equalsWith(string1, pattern)
    r = strcmp(string1,pattern);
end

function [cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx] = addContent(cbContentLines,cbContentLinesFull,cbContentLinesIdx,cbContentLinesFullIdx,lineIdx,both)
    if cbContentLinesFullIdx > length(cbContentLinesFull)
        cbContentLinesFull = [cbContentLinesFull -ones(1,100)];
    end
    cbContentLinesFull(cbContentLinesFullIdx) = lineIdx;
    cbContentLinesFullIdx = cbContentLinesFullIdx + 1;

    if both
        if cbContentLinesIdx > length(cbContentLines)
            cbContentLines = [cbContentLines -ones(1,100)];
        end
        cbContentLines(cbContentLinesIdx) = lineIdx;
        cbContentLinesIdx = cbContentLinesIdx + 1;
    end
end

function setBlockData(blockMap,blockId,contentLines,conn,connFill,removeLastContentLine)
    if ~blockMap.isKey(blockId)
        return; % a block without nodes. shouldn't happen.
    end
    theblock = blockMap(blockId);
    if removeLastContentLine
        theblock.contentLines = [theblock.contentLines(1:end-1) contentLines];
    else
        theblock.contentLines = [theblock.contentLines contentLines];
    end    
    theblock.conn = conn;
    theblock.connFill = connFill;
    blockMap(blockId) = theblock; %#ok<NASGU>
end

function [newBlock,conn,connFill] = getBlockData(blockMap,blockId)
    if blockMap.isKey(blockId)
        newBlock = false;
        theblock = blockMap(blockId);
        conn = theblock.conn;
        connFill = theblock.connFill;
    else
        newBlock = true;
        conn = {};
        connFill = {};
        
        s = struct();
        s.contentLines = [];
        s.conn = conn;
        s.connFill = connFill;
        
        blockMap(blockId) = s; %#ok<NASGU>
    end
end

function [conn] = addConnection(nodeId1, nodeId2, conn)
    conn{1,end+1} = nodeId1; % from
    conn{2,end}   = nodeId2; % to
end

function [am,idx2idArray,edge2idxMat,connCount,total] = buildAdjacencyMatrix(conn)
    if isempty(conn)
        am = [];
        idx2idArray = [];
        edge2idxMat = [];
    else
        from = conn(1,:); % from nodes
        to = conn(2,:); % to nodes
        [idx2idArray,~,ic] = unique([from to]);

        fromIdx = ic(1:length(from));
        toIdx = ic(length(from)+1:end);
        edge2idxMat = [fromIdx' ; toIdx'];

        nodeCount = max(ic);
        am = false(nodeCount); % adjacency matrix

        idx1 = sub2ind(size(am),fromIdx,toIdx);
        idx2 = sub2ind(size(am),toIdx,fromIdx);
        idxD = sub2ind(size(am),1:nodeCount,1:nodeCount);
        am(idx1) = true;
        am(idx2) = true;
        am(idxD) = false; % diagonal
    end

    connCount = sum(am,1);
    total = sum(connCount,2);
end

function printLines(fileId,am,idx2idArray,connCount,total)
    if total == 0
        return;
    end

    fprintf(fileId, 'N\n');

    [~,sidx] = sort(connCount);
    for ni = sidx
        firstNode = -1;
        first = true;
        search = true;
        node = ni;

        while(search)
            neighbours = find(am(node,:));
            search = false;
            for nni = neighbours
                if ~am(node,nni)
                    continue; % edge visited
                end
                if first
                    fprintf(fileId, '%sM\n', cell2mat(idx2idArray(node)));
                    first = false;
                    firstNode = node;
                end
                am(node,nni) = false;
                am(nni,node) = false;
                if nni == firstNode
                    % closed path (polygon) -> use a 'closepath' command instead of a line
                    fprintf(fileId, 'cp\n');
                else
                    fprintf(fileId, '%sL\n', cell2mat(idx2idArray(nni)));
                end
                node = nni;
                search = true;
                break;
            end
        end
    end
    
    fprintf(fileId, 'S\n');
end

function printFills(fileId,am,idx2idArray,total,edge2idxMat)
    if total == 0
        return;
    end
    
    edgepolymat = zeros(size(am));
    edgeusemat = zeros(size(am));
    
    nodeCount = size(idx2idArray,2);
    edgeCount = size(edge2idxMat,2);
    polyIdxs = zeros(1,edgeCount);

    % determine connections -> polygon:
    polyIdx = 0;
    edge = 1;
    while true
        if edge <= edgeCount
            startIdx = edge2idxMat(1,edge);
        else
            break;
        end
        polyIdx = polyIdx + 1;
        
        while edge <= size(edge2idxMat,2)
            tidx = edge2idxMat(2,edge);
            polyIdxs(edge) = polyIdx;
            
            edge = edge + 1;
            if startIdx == tidx
                break; % polygon finished
            end
        end
    end
    
    % check whether or not a polygon has the same edge defined twice
    polyCount = polyIdx;
    selfEdges = false(1,polyCount);
    for ii = 1:polyCount
        selfEdges(ii) = hasEdgeWithItself(edge2idxMat,polyIdxs,ii);
    end    
    
    % check if there are initial self edges and if so, just pretend we have been visiting those polygons already:
    k=find(selfEdges);
    for kk = k
        ii = edge2idxMat(:,polyIdxs == kk);
        idxs1 = sub2ind(size(edgeusemat), ii(1,:), ii(2,:));
        idxs2 = sub2ind(size(edgeusemat), ii(2,:), ii(1,:));
        idxs = [idxs1 idxs2];
        edgeusemat(idxs) = edgeusemat(idxs) + 1;
        edgeusemat(idxs) = edgeusemat(idxs) + 1;
        edgepolymat(idxs) = kk;
        edgepolymat(idxs) = kk;
    end
    
    
    polyIdx = 0;
    edge = 1;
    initialEdgeCount = size(edge2idxMat,2);
    while true
        if edge <= initialEdgeCount
            startIdx = edge2idxMat(1,edge);
        else
            break;
        end
        polyIdx = polyIdx + 1;
        
        if selfEdges(polyIdx)
            % polygon has edge with itself, don't try to merge and skip polygon instead
            edge = edge + find(edge2idxMat(2,edge:end) == tidx,1);
        else
            handledPolyMap = containers.Map('KeyType','double','ValueType','any');

            while edge <= initialEdgeCount
                fidx = edge2idxMat(1,edge);
                tidx = edge2idxMat(2,edge);
                                
                removeEdge = false;
                nPolyIdx = edgepolymat(fidx,tidx);
                if nPolyIdx > 0
                    if ~selfEdges(nPolyIdx)
                        if handledPolyMap.isKey(nPolyIdx)
                            % leave the edge intact, except if it's connected to the shared edge
                            val = handledPolyMap(nPolyIdx);
                            f = val(1);
                            t = val(2);
                            connected = true;
                            if f == fidx
                                f = tidx;
                            elseif f == tidx
                                f = fidx;
                            elseif t == fidx
                                t = tidx;
                            elseif t == tidx
                                t = fidx;
                            else
                                connected = false;
                            end
                            if connected
                                fusage = sum(edgeusemat(fidx,:) > 0);
                                tusage = sum(edgeusemat(tidx,:) > 0);
                                removeEdge = (fusage == 1 || tusage == 1);
                                if removeEdge
                                    handledPolyMap(nPolyIdx) = [f t];
                                end
                            end
                        else
                            % remove the first common shared edge
                            handledPolyMap(nPolyIdx) = [fidx tidx];
                            removeEdge = true;
                        end
                    end
                else
                    edgepolymat(fidx,tidx) = polyIdx;
                    edgepolymat(tidx,fidx) = polyIdx;
                end
                
                if removeEdge
                    edgepolymat(fidx,tidx) = 0;
                    edgepolymat(tidx,fidx) = 0;
                    edgeusemat(fidx,tidx) = 0;
                    edgeusemat(tidx,fidx) = 0;
                    polyIdxs(edge) = 0;
                else
                    edgeusemat(fidx,tidx) = edgeusemat(fidx,tidx) + 1;
                    edgeusemat(tidx,fidx) = edgeusemat(tidx,fidx) + 1;
                end
                
                edge = edge + 1;
                if startIdx == tidx
                    break; % polygon finished
                end
            end

            % merge all handled polygons:
            for k = cell2mat(handledPolyMap.keys())
                edgepolymat(edgepolymat == k) = polyIdx;
                polyIdxs(polyIdxs == k) = polyIdx;
            end
            selfEdges(polyIdx) = hasEdgeWithItself(edge2idxMat,polyIdxs,polyIdx);
        end
    end
    
        
    
    connCount = sum(edgeusemat, 1);

    coordinates = zeros(nodeCount,2);
    remainingNodes = find(connCount);
    for c = remainingNodes
        coordinates(c,:) = extractCoords(idx2idArray(c));
    end

    fprintf(fileId, 'N\n');

    [~,sidx] = sort(connCount); % sort by lowest connection count
    for ni = sidx
        firstNode = -1;
        prevNode = -1;
        first = true;
        search = true;
        node = ni;
        unkLeftRight = 0;

        while(search)
            c = edgeusemat(node,:);
            [~,sidx2] = sort(c(c>0),'descend'); % sort by edge-usage (select higher usage first)
            neighbours = find(c);
            neighbours = neighbours(sidx2);
            neighbours(neighbours == prevNode) = []; % don't go backwards
            search = false;
            nidx = 0;
            for nni = neighbours
                nidx = nidx + 1;
                if edgeusemat(node,nni) == 0
                    continue; % edge already visited
                end
                
                if length(neighbours) >= 2
                    if unkLeftRight > 0
                        p = coordinates(prevNode,:);
                        c = coordinates(node,:);
                        n = coordinates(nni,:);
                        
                        valid = true;
                        for nni2 = neighbours
                            if nni2 == nni
                                continue;
                            end
                            
                            a = coordinates(nni2,:);
                            leftRight = isNodeRight(p,c,n,a);

                            if unkLeftRight ~= leftRight
                                valid = false;
                                break;
                            end
                        end
                        
                        if ~valid
                            continue; % other neighbour
                        end
                    elseif edgeusemat(node,nni) == 2 && prevNode ~= -1
                        % a double edge with more than one option -> remember which way we go (ccw or cw)
                        p = coordinates(prevNode,:); % previous node
                        c = coordinates(node,:); % current node
                        n = coordinates(nni,:); % next node
                        a = coordinates(neighbours(1 + ~(nidx-1)),:); % alternative node
                        
                        unkLeftRight = isNodeRight(p,c,n,a);
                    end
                end
                
                if first
                    fprintf(fileId, '%sM\n', cell2mat(idx2idArray(node)));
                    first = false;
                    firstNode = node;
                end
                
                edgeusemat(node,nni) = edgeusemat(node,nni) - 1;
                edgeusemat(nni,node) = edgeusemat(nni,node) - 1;
                if nni == firstNode
                    % closed path (polygon) -> use a 'closepath' command instead of a line
                    fprintf(fileId, 'cp\n');
                else
                    fprintf(fileId, '%sL\n', cell2mat(idx2idArray(nni)));
                end
                prevNode = node;
                node = nni;
                search = true;
                break;
            end
        end
    end
    
    fprintf(fileId, 'f\n');
end

function value = hasEdgeWithItself(id2idxMat,polyIdxs,polyIdx)
    % check if same edge exists twice in polygon
    edgePoly = id2idxMat(:,polyIdxs == polyIdx);
    edgePoly2 = [edgePoly(2,:) ; edgePoly(1,:)];
    [~,~,ic] = unique([edgePoly' ; edgePoly2'],'rows');
    ic = accumarray(ic,1); % count the number of identical elements
    value = any(ic > 1);
end

function leftRight = isNodeRight(p,c,n,a)
    v1 = c - p; v1 = v1 ./ norm(v1);
    v2 = n - c; v2 = v2 ./ norm(v2);
    v3 = a - c; v3 = v3 ./ norm(v3);

    s2 = sign(v2(1) * v1(2) - v2(2) * v1(1));
    side = s2 - sign(v3(1) * v1(2) - v3(2) * v1(1));
    if side == 0
        % both vectors on the same side
        if s2 == 1
            % both vectors left
            right = dot(v1,v2) > dot(v1,v3);
        else
            % both vectors right
            right = dot(v1,v2) < dot(v1,v3);
        end
    else
        right = side < 0;
    end
    
    leftRight = 1;
    if right
        leftRight = 2;
    end
end

function p = extractCoords(nodeId)
    nodeId = cell2mat(nodeId);
    k = strfind(nodeId, ' ');
    x = str2double(nodeId(1:k(1)));
    y = str2double(nodeId(k(1)+1:end));
    p = [x y];
end

function writeBlocks(blockList, blockMap, fileId, fileContent)
    for ii = 1:length(blockList)
        blockId = blockList(ii).prefix;
        fprintf(fileId, 'GS\n%s', blockId);
        
        theblock = blockMap(blockId);
        contentLines = theblock.contentLines;

        % build adjacency matrix from connections:
        [amL,idx2idArrayL,~,connCountL,totalL] = buildAdjacencyMatrix(theblock.conn);
        [amF,idx2idArrayF,edge2idxMatF,~,totalF] = buildAdjacencyMatrix(theblock.connFill);
        
        total = totalL + totalF;

        if total == 0
            if ~isempty(contentLines)
                if isempty(regexp(blockId, sprintf('clip\n$'), 'once')) % prefix does not end with clip
                    fprintf(fileId, 'N\n');
                end

                fprintf(fileId, '%s\n', strjoin(fileContent(contentLines),'\n'));
            end
        else
            printLines(fileId,amL,idx2idArrayL,connCountL,totalL);
            printFills(fileId,amF,idx2idArrayF,totalF,edge2idxMatF);

            if ~isempty(contentLines)
                fprintf(fileId, '%s\n', strjoin(fileContent(contentLines),'\n'));
            end
        end

        fprintf(fileId, 'GR\n');
    end
end


function [b] = r2(x, y)

x = x;
y = y;

lengthx = length(x);

xmean = mean(x);
ymean = mean(y);

xcov = zeros(1,length(x));
ycov = zeros(1,length(y));

for i = 1:lengthx;
    xcov(i) = x(i) - xmean;
end

for i = 1:lengthx;
    ycov(i) = y(i) - ymean;
end

xcov = transpose (xcov);
ycov = transpose (ycov);

mult = xcov.*ycov;
numerator = sum(mult);

xcov2 = xcov.*xcov;
sumxcov2 = sum(xcov2);
ycov2 = ycov.*ycov;
sumycov2 = sum(ycov2);
mult2 = sumxcov2*sumycov2;
denominator = sqrt(mult2);

r = numerator/denominator;

b = r*r;

end

function [p,V] = kuipertest2c(x1,x2)


x1 = sort(x1);
x2 = sort(x2);
n1 = length(x1(~isnan(x1)));
n2 = length(x2(~isnan(x2)));

binEdges    =  [-inf ; sort([x1;x2]) ; inf];

binCounts1  =  histc (x1 , binEdges, 1);
binCounts2  =  histc (x2 , binEdges, 1);

sumCounts1  =  cumsum(binCounts1)./sum(binCounts1);
sumCounts2  =  cumsum(binCounts2)./sum(binCounts2);

sampleCDF1  =  sumCounts1(1:end-1);
sampleCDF2  =  sumCounts2(1:end-1);

deltaCDF1  =  sampleCDF2 - sampleCDF1;
maxdeltaCDF1 = max(deltaCDF1);


deltaCDF2  =  sampleCDF1 - sampleCDF2;
maxdeltaCDF2 = max(deltaCDF2);

V = maxdeltaCDF1 + maxdeltaCDF2;

ne = ((n1*n2)/(n1+n2));
lambda  =  max((sqrt(ne) + 0.155 + (0.24/sqrt(ne))) * V);

if lambda<0.4  
p=1;  
h=0;
return
end

j=(1:100)';
pare=4*lambda*lambda*(j.^2)-1;
expo=exp(-2*lambda*lambda*(j.^2));
argo=pare.*expo;
p=2*sum(argo);

p = p;
V = V;

end











function [Y,stress,disparities] = mdscale_new(D,p,varargin)
%MDSCALE Non-Metric and Metric Multidimensional Scaling.
%   Y = MDSCALE(D,P) performs non-metric multidimensional scaling on the
%   N-by-N dissimilarity matrix D, and returns Y, a configuration of N
%   points (rows) in P dimensions (cols).  The Euclidean distances between
%   points in Y approximate a monotonic transformation of the corresponding
%   dissimilarities in D.  By default, MDSCALE uses Kruskal's normalized
%   STRESS1 criterion.
%
%   You can specify D as either a full N-by-N matrix, or in upper triangle
%   form such as is output by PDIST.  A full dissimilarity matrix must be
%   real and symmetric, and have zeros along the diagonal and non-negative
%   elements everywhere else.  A dissimilarity matrix in upper triangle
%   form must have real, non-negative entries.  MDSCALE treats NaNs in D as
%   missing values, and ignores those elements.  Inf is not accepted.
%
%   You can also specify D as a full similarity matrix, with ones along the
%   diagonal and all other elements less than one.  MDSCALE transforms a
%   similarity matrix to a dissimilarity matrix in such a way that
%   distances between the points returned in Y approximate sqrt(1-D).  To
%   use a different transformation, transform the similarities prior to
%   calling MDSCALE.
%
%   [Y,STRESS] = MDSCALE(D,P) returns the minimized stress, i.e., the
%   stress evaluated at Y.
%
%   [Y,STRESS,DISPARITIES] = MDSCALE(D,P) returns the disparities, i.e. the
%   monotonic transformation of the dissimilarities D.
%
%   [...] = MDSCALE(..., 'PARAM1',val1, 'PARAM2',val2, ...) allows you to
%   specify optional parameter name/value pairs that control further details
%   of MDSCALE.  Parameters are:
%
%   'Criterion' - The goodness-of-fit criterion to minimize.  This also
%       determines the type of scaling, either non-metric or metric, that
%       MDSCALE performs.  Choices for non-metric scaling are:
%
%           'stress'  - Stress normalized by the sum of squares of
%                       the interpoint distances, also known as STRESS1.
%                       This is the default.
%           'sstress' - Squared Stress, normalized with the sum of 4th
%                       powers of the interpoint distances.
%
%       Choices for metric scaling are:
%
%           'metricstress'  - Stress, normalized with the sum of squares
%                             of the dissimilarities.
%           'metricsstress' - Squared Stress, normalized with the sum of
%                             4th powers of the dissimilarities.
%           'sammon'        - Sammon's nonlinear mapping criterion.
%                             Off-diagonal dissimilarities must be
%                             strictly positive with this criterion.
%           'strain'        - A criterion equivalent to that used in
%                             classical MDS.
%
%   'Weights' - A matrix or vector the same size as D, containing
%       nonnegative dissimilarity weights.  You can use these to weight the
%       contribution of the corresponding elements of D in computing and
%       minimizing stress.  Elements of D corresponding to zero weights are
%       effectively ignored.  Note: when you specify weights as a full matrix,
%       its diagonal elements are ignored and have no effect, since the
%       corresponding diagonal elements of D do not enter into the stress
%       calculation.
%
%   'Start' - Method used to choose the initial configuration of points
%       for Y.  Choices are:
%
%       'cmdscale' - Use the classical MDS solution.  This is the default.
%                    'cmdscale' is not valid when there are zero weights.
%       'random'   - Choose locations randomly from an appropriately
%                    scaled P-dimensional normal distribution with
%                    uncorrelated coordinates.
%       matrix     - An N-by-P matrix of initial locations.  In this
%                    case, you can pass in [] for P, and MDSCALE infers P
%                    from the second dimension of the matrix. You can also
%                    supply a 3D array, implying a value for 'Replicates'
%                    from the array's third dimension.
%
%   'Replicates' - Number of times to repeat the scaling, each with a new
%       initial configuration.  Defaults to 1.
%
%   'Options' - Options for the iterative algorithm used to minimize the
%       fitting criterion, as created by STATSET.  Choices of STATSET
%       parameters are:
%
%       'Display'     - Level of display output.  Choices are 'off' (the
%                       default), 'iter', and 'final'.
%       'MaxIter'     - Maximum number of iterations allowed.  Defaults
%                       to 200.
%       'TolFun'      - Termination tolerance for the stress criterion
%                       and its gradient.  Defaults to 1e-4.
%       'TolX'        - Termination tolerance for the configuration
%                       location step size.  Defaults to 1e-4.
%
%   Example:
%
%      % Load cereal data, and create a dissimilarity matrix.
%      load cereal.mat
%      X = [Calories Protein Fat Sodium Fiber Carbo Sugars Shelf Potass Vitamins];
%      X = X(Mfg == 'K',:); % take a subset from a single manufacturer
%      dissimilarities = pdist(X);
%
%      % Use non-metric scaling to recreate the data in 2D, and make a
%      % Shepard plot of the results.
%      [Y,stress,disparities] = mdscale(dissimilarities,2);
%      distances = pdist(Y);
%      [dum,ord] = sortrows([disparities(:) dissimilarities(:)]);
%      plot(dissimilarities,distances,'bo', ...
%           dissimilarities(ord),disparities(ord),'r.-');
%      xlabel('Dissimilarities'); ylabel('Distances/Disparities')
%      legend({'Distances' 'Disparities'}, 'Location','NorthWest');
%
%      % Do metric scaling on the same dissimilarities.
%      [Y,stress] = mdscale(dissimilarities,2,'criterion','metricsstress');
%      distances = pdist(Y);
%      plot(dissimilarities,distances,'bo', ...
%           [0 max(dissimilarities)],[0 max(dissimilarities)],'k:');
%      xlabel('Dissimilarities'); ylabel('Distances')
%
%   See also CMDSCALE, PDIST, STATSET.

%   In non-metric scaling, MDSCALE finds a configuration of points whose
%   pairwise Euclidean distances have approximately the same rank order as
%   the corresponding dissimilarities.  Equivalently, MDSCALE finds a
%   configuration of points, whose pairwise Euclidean distances approximate
%   a monotonic transformation of the dissimilarities.  These transformed
%   values are known as the disparities.
%
%   In metric scaling, MDSCALE finds a configuration of points whose pairwise
%   Euclidean distances approximate the dissimilarities directly.  There are
%   no disparities in metric scaling.
%
%   References:
%      [1] Cox, R,.F. and Cox, M.A.A. (1994) Multidimensional Scaling,
%          Chapman&Hall.
%      [2] Davison, M.L. (1983) Multidimensional Scaling, Wiley.
%      [3] Seber, G.A.F., (1984) Multivariate Observations, Wiley.

%   Copyright 1993-2012 The MathWorks, Inc.


if nargin > 2
    [varargin{:}] = convertStringsToChars(varargin{:});
end

if nargin < 2
    error(message('stats:mdscale:TooFewInputs'));
end

paramNames = {'criterion' 'weights' 'start' 'replicates' 'options'};
paramDflts = {[] [] [] [] []};
[criterion,weights,start,nreps,options] = ...
                           internal.stats.parseArgs(paramNames, paramDflts, varargin{:});

D = double(D);
[n,m] = size(D);

% Make sure weights match dissimilarities, and are non-negative.
if ~isempty(weights)
    if isequal(size(weights),size(D))
        if any(weights < 0)
            error(message('stats:mdscale:NegativeWeights'));
        end
    else
        error(message('stats:mdscale:InputSizeMismatch'));
    end
    weighted = true;
else
    weighted = false;
end

% Treat NaNs as missing, and zero out the corresponding weights.
missing = find(isnan(D));
if ~isempty(missing)
    if ~weighted
        weights = ones(size(D), class(D));
        weighted = true;
    end
    weights(missing) = 0;
end

isEmptyOrZeroWgt = @(badD) (~weighted && isempty(badD)) || ...
                           (weighted && all(weights(badD) == 0));
% Lower triangle form for D, make sure it's a valid dissimilarity matrix
if n == 1
    n = ceil(sqrt(2*m)); % (1+sqrt(1+8*m))/2, but works for large m
    badD = find((D < 0) | ~isfinite(D));
    if (n*(n-1)/2 == m) && isEmptyOrZeroWgt(badD)
        dissimilarities = D;
    else
        error(message('stats:mdscale:InvalidDissimilarity'));
    end
    fullInputD = false;

% Full matrix form, make sure it's valid similarity/dissimilarity matrix
elseif n == m
    badD = find((D < 0) | ~isfinite(D) | abs(D-D') > 10*eps*max(D(:)));
    if isEmptyOrZeroWgt(badD)
        
        % It's a dissimilarity matrix
        if all(diag(D) == 0)
            % nothing to do
            
        % It's a similarity matrix -- transform to dissimilarity matrix.
        % the sqrt is not entirely arbitrary, see Seber, eqn. 5.73
        else
            badD = find(D > 1);
            if all(diag(D) == 1) && isEmptyOrZeroWgt(badD)
                D = sqrt(1 - D);
            else
                error(message('stats:mdscale:InvalidDissimilarities'))
            end
        end
        fullInputD = true;
        
        % Get the lower triangle form for the dissimilarities and weights
        dissimilarities = D(tril(true(size(D)),-1))';
        if weighted
            % This throws away the diagonal terms of a full weight matrix.
            % They are not needed because the on-diagonal dissimilarities do
            % not enter into computation of the fit criteria, except for
            % strain, which needs unit weights on the diagonal -- those will
            % be created below.
            weights = weights(tril(true(size(D)),-1))';
        end
    else
        error(message('stats:mdscale:InvalidDissimilarities'))
    end
    
else
    error(message('stats:mdscale:InvalidDissimilarities'))
end
if weighted
    zeroWgts = find(weights == 0);
    % Fill dissimilarities corresponding to zero weights with anything
    % finite: these dissimilarities must get ignored by being multiplied by
    % the zero weights.
    if ~isempty(zeroWgts)
        dissimilarities(zeroWgts) = 0;
    end
    % It's not strictly necessary to do this for nonmetric scaling, because
    % the dissimilarities get sent through lsqisotonic.
end

% Use Kruskal's STRESS1 by default, or whatever is specified.
if isempty(criterion)
    stressFun = @stressCrit;
    metric = false; strain = false;
elseif ischar(criterion)
    funNames = {'stress','sstress','metricstress','metricsstress','sammon','strain'};
    [~,i] = internal.stats.getParamVal(criterion,funNames,'Criterion');
    if length(i) > 1
        error(message('stats:mdscale:AmbiguousCriterion', criterion));
    elseif isempty(i)
        error(message('stats:mdscale:UnknownCriterion', criterion));
    end
    switch i
    case 1 % 'stress'
        stressFun = @stressCrit;
        metric = false; strain = false;
    case 2 % 'sstress'
        stressFun = @sstressCrit;
        metric = false; strain = false;
    case 3 % 'metricstress'
        stressFun = @metricStressCrit;
        metric = true; strain = false;
    case 4 % 'metricsstress'
        stressFun = @metricSStressCrit;
        metric = true; strain = false;
    case 5 % 'sammon'
        if any(dissimilarities <= 0)
            error(message('stats:mdscale:NonpositiveDissimilarities'));
        end
        stressFun = @sammonCrit;
        metric = true; strain = false;
    case 6 % 'strain'
        stressFun = @strainCrit;
        metric = true; strain = true;
    end
else
    error(message('stats:mdscale:InvalidStressFun'));
end

% Use the classical solution as a starting point by default.
if isempty(start) || ...
   (ischar(start) && isequal(strfind('cmdscale', lower(start)),1))
    if weighted && ~isempty(zeroWgts)
        error(message('stats:mdscale:NeedPosWeights'));
    end
    % No sense in replicates if there's only one starting point.
    nreps = 1;
    start = 'cmdscale';
    
% Use a random configuration as a starting point.
elseif ischar(start) && isequal(strfind('random', lower(start)),1)
    start = 'random';
    
    % Scale random starting locations to have an average squared distance
    % equal to the average squared dissimilarity in D.
    if weighted
        sigsq = mean(dissimilarities(weights>0).^2)./(2.*p);
    else
        sigsq = mean(dissimilarities.^2)./(2.*p);
    end
    
% User-supplied configuration(s) as a starting point.
elseif isnumeric(start)
    [r,c,pg] = size(start);
    
    % Infer the number of replicates from the number of starting
    % configurations supplied.
    if isempty(nreps)
        nreps = pg;
    end
    % Otherwise, will have to verify that the number of starting
    % configurations supplied equals the number of replicates.
    
    % The number of dimensions, p, can be left out if 'start' is given
    % explicitly.  Infer p from the starting configuration(s) if necessary.
    if isempty(p), p = c; end
    
    % Make sure the starting configuration(s) have the right size, and save
    % them for later.
    if (r==n) && (c==p) && (pg==nreps)
        explicitY0 = start;
        start = 'explicit';
    else
        error(message('stats:mdscale:InputSizeStart'));
    end
    
else
    error(message('stats:mdscale:InvalidStart'));
end

% Assume one replicate if it was not given or inferred from start.
if isempty(nreps), nreps = 1; end

%
% Done processing input args, begin calculation
%

if strain
    % Strain uses transformed dissimilarities, and needs them as a
    % square matrix.
    A = squareform(-0.5 .* (dissimilarities.^2));

    if weighted
        % The strain criterion needs to include the diagonal terms of A in the
        % computation, so we add unit diagonal weights.  If the weights were
        % originally a full matrix, the original on-diag weights have already
        % been dropped, this replaces them.
        weights = squareform(weights) + eye(size(A));
        % In addition to per-dissimilarity weights, strain needs per-point
        % weights in order to compute a weighted mean of the configuration.
        obsWeights = sum(weights,2) - diag(weights); % ignore diagonal terms
        obsWeights = obsWeights ./ sum(obsWeights);
    else
        weights = 1;
        obsWeights = 1 / n;
    end
    
    strainFun = @(Y, A, weights) strainCrit(Y, A, weights, obsWeights);
end

bestStress = Inf; bestY = []; bestDisparities = [];
for rep = 1:nreps
    % Initialize the configuration of points.
    switch start
    case 'cmdscale'
        Y0 = cmdscale(D); % the original D, full if given that way
        if size(Y0,2) >= p
            Y0 = Y0(:,1:p);
        else % D had more than (n-p) negative eigenvalues
            warning(message('stats:mdscale:ZeroPad'));
            Y0 = [Y0 zeros(n,p-size(Y0,2),class(Y0))];
        end
    case 'random'
        Y0 = cast(randn(n,p),class(dissimilarities)) .* sqrt(sigsq);
    case 'explicit'
        Y0 = explicitY0(:,:,rep);
    end
    
    % Do metric or non-metric multidimensional scaling.
    if metric
        if strain
            [Y,stress] = MDS(Y0,A,weights,strainFun,metric,weighted,options);
        else
            [Y,stress] = MDS(Y0,dissimilarities,weights,stressFun,metric,weighted,options);
        end
        if nargout > 2
            disparities = dissimilarities;
        end
    else
        [Y,stress,~,disparities] = MDS(Y0,dissimilarities,weights,stressFun,metric,weighted,options);
    end
    
    % Save this solution if it's the best one so far.
    if stress < bestStress
        bestStress = stress;
        bestY = Y;
        if nargout > 2, bestDisparities = disparities; end
    end
end

% Remember the best solution.
stress = bestStress;
Y = bestY;
if nargout > 2
    if fullInputD
        disparities = squareform(bestDisparities);
    else
        disparities = bestDisparities;
    end
end

% Rotate the solution to principal component axes.
[~,score] = pca(Y,'Economy',false);
Y = score;

% Enforce a sign convention on the solution -- the largest element
% in each coordinate will have a positive sign.
[~,maxind] = max(abs(Y),[],1);
d = size(Y,2);
colsign = sign(Y(maxind + (0:n:(d-1)*n)));
Y = bsxfun(@times,Y,colsign);
end

%==========================================================================

function [Y,stress,iter,disparities] = ...
             MDS(Y,dissimilarities,weights,stressFun,metric,weighted,options)

n = size(Y,1);

% Merge default and user options.
options = statset(statset('mdscale'), options);

% Start with a loose tolerance in the line search.
lineSearchOpts = ...
    struct('Display','off', 'MaxFunEvals',100, 'MaxIter',100, 'TolX',1e-3);

[~,verb] = internal.stats.getParamVal(options.Display, ...
    ['off   '; 'notify';  'final '; 'iter  '],'Options.Display');
verb = verb-1;
if verb > 2
    fprintf('\n');
    fprintf('                     Stress          Norm of         Norm of     Line Search\n');
    fprintf('   Iteration       Criterion        Gradient           Step       Iterations\n');
    fprintf('  ---------------------------------------------------------------------------\n');
end

% For metric scaling, there are no disparities, but for convenience in
% the code, set them equal to the dissimilarities.
if metric
    disparities = dissimilarities;
end
if ~weighted
    weights = 1; % dummy weights for the criterion functions
end

iter = 0;
resetCG = true;
oldStress = NaN;
stepLen = NaN;

% Initialize this variable so it will not appear to be a function here
oldNormGrad = 0;

while true
    % Center Y: Stress is invariant to location, and this keeps the
    % configuration from wandering.
    Y = Y - repmat(mean(Y,1),n,1);
    
    % For non-metric scaling, compute disparities as the values closest to
    % the current interpoint distances, in the least squares sense, while
    % constrained to be monotonic in the given dissimilarities.
    if ~metric
        distances = pdist(Y);
        
        % Keep the configuration on the same scale as the dissimilarities.
        % The non-metric forms of Stress are invariant to scale.
        scale = max(dissimilarities)/max(distances);
        Y = Y * scale;
        distances = distances * scale;

        if weighted
            disparities = lsqisotonic(dissimilarities, distances, weights);
        else
            disparities = lsqisotonic(dissimilarities, distances);
        end
    end
    
    % Compute stress for the current configuration, and its gradient with
    % respect to Y, with disparities held constant.
    [stress,grad] = feval(stressFun, Y, disparities, weights);
    normGrad = norm(grad(:));
    
    if verb > 2
        if iter == 0
            fprintf('      %6d    %12g    %12g\n',iter,stress,normGrad);
        else
            fprintf('      %6d    %12g    %12g    %12g          %6d\n', ...
                    iter,stress,normGrad,stepLen,output.funcCount);
        end
    end
    
    % Test for convergence or failure.
    if stress < options.TolFun
        % The current configuration might fit the dissimilarities exactly, in
        % which case the gradient is not necessarily small, since we're at the
        % lower limit of stress, not a local minimum.
        if verb > 2
            fprintf('%s\n',getString(message('stats:mdscale:TerminatedCriterion')));
        end
        break;
    elseif normGrad < options.TolFun*stress
        if verb > 2
            fprintf('%s\n',getString(message('stats:mdscale:TerminatedRelativeNormOfGradient')));
        end
        break;
    elseif (oldStress-stress) < options.TolFun*stress
        if verb > 2
            fprintf('%s\n',getString(message('stats:mdscale:TerminatedRelativeChangeInCriterion')));
        end
        break;
    elseif stepLen < options.TolX * norm(Y(:))
        if verb > 2
            fprintf('%s\n',getString(message('stats:mdscale:TerminatedNormOfChangeInConfiguration')));
        end
        break;
    elseif iter == options.MaxIter
        warning(message('stats:mdscale:IterOrEvalLimit'));
        break;
    end
    
    % Use Polak-Riviere to compute the CG search direction.
    if resetCG
        resetCG = false;
        stepDir = -grad;
    else
        beta = max(((grad(:)-oldGrad(:))'*grad(:)) / oldNormGrad^2, 0);
        stepDir = -grad + beta*stepDir;
    end
    
    oldStress = stress;
    oldGrad = grad;
    oldNormGrad = normGrad;
    
    % Do a line search to minimize stress in the CG search direction. First
    % find an upper bound on step length, at which the stress is higher, then
    % search between zero step length and that.
    maxStepLen = 2;
    while true
        stress = lineSearchCrit(maxStepLen,Y,stepDir,disparities,weights,stressFun);
        if stress > oldStress, break; end
        maxStepLen = 2*maxStepLen;
    end
    [alpha, stress, err, output] = ...
            fminbnd(@lineSearchCrit,0,maxStepLen,lineSearchOpts, ...
                    Y,stepDir,disparities,weights,stressFun);
    if (err == 0)
        warning(message('stats:mdscale:LineSrchIterLimit'));
    elseif (stress > oldStress)
        % FMINBND occasionally finds a local minimum that is higher than the
        % previous stress, because the stress initially decreases to the true
        % minimum, then increases and has a local min beyond that.  Have no
        % truck with that.
        while true
            alpha = alpha/2;
            if alpha <= 1e-12
                error(message('stats:mdscale:NoSolution'));
            end
            stress = lineSearchCrit(alpha,Y,stepDir,disparities,weights,stressFun);
            if stress < oldStress, break; end
        end
        resetCG = true;
    elseif (err < 0) % should never happen
        error(message('stats:mdscale:NoSolution'));
    end
    
    % Take the downhill step.
    Ystep = alpha*stepDir;
    stepLen = alpha*norm(stepDir);
    Y = Y + Ystep;
    iter = iter + 1;
    
    % Tighten up the line search tolerance, but not beyond the requested
    % tolerance.
    lineSearchOpts.TolX = max(lineSearchOpts.TolX/2, options.TolX);
end
if verb > 1
    fprintf('%s\n',getString(message('stats:mdscale:IterationsStress',iter,sprintf('%g',stress))));
end
end
%==========================================================================

function val = lineSearchCrit(t,Y,stepDir,disparities,weights,stressFun)
%LINESEARCHOBJFUN Objective function for linesearch in MDS.

% Given a step size, evaluate the stress at a downhill step away from the
% current configuration.
val = feval(stressFun,Y+t*stepDir,disparities,weights);
end

%==========================================================================

function [S,grad] = stressCrit(Y,disparities,weights)
%STRESS Stress criterion for nonmetric multidimensional scaling.

% The Euclidean norm of the differences between the distances and the
% disparities, normalized by the Euclidean norm of the distances.
%
% Zero weights are ok as long as the corresponding dissimilarities are
% finite.  However, any zero distances will throw an error:  this criterion
% is not differentiable when two points are coincident - it involves, in
% effect, abs(Y(i,k)-Y(j,k)).  This does sometimes happen, even though a
% configuration that is a local minimizer cannot have coincident points.

distances = pdist(Y);
diffs = distances - disparities;
sumDiffSq = sum(weights.*diffs.^2);
sumDistSq = sum(weights.*distances.^2);
S = sqrt(sumDiffSq ./ sumDistSq);
if nargout > 1
    [n,p] = size(Y);
    grad = zeros(n,p);
    if sumDiffSq > 0
        if all(distances > 0)
            dS = squareform(weights.* ...
                     (diffs./sumDiffSq - distances./sumDistSq) ./ distances);
            repcols = zeros(1,n);
            for i = 1:p
                repcols = repcols + 1;
                dY = Y(:,repcols) - Y(:,repcols)';
                grad(:,i) = sum(dS.*dY,2) .* S;
            end
        else
            error(message('stats:mdscale:ColocatedPoints'));
        end
    end
end
end

%==========================================================================

function [S,grad] = metricStressCrit(Y,dissimilarities,weights)
%METRICSTRESS Stress criterion for metric MDS.

% The Euclidean norm of the differences between the distances and the
% dissimilarities, normalized by the Euclidean norm of the dissimilarities.
%
% Zero weights are ok as long as the corresponding dissimilarities are
% finite.  However, any zero distances will throw an error:  this criterion
% is not differentiable when two points are coincident - it involves, in
% effect, abs(Y(i,k)-Y(j,k)).  This does sometimes happen, even though a
% configuration that is a local minimizer cannot have coincident points.

distances = pdist(Y);
diffs = distances - dissimilarities;
sumDiffSq = sum(weights.*diffs.^2);
sumDissSq = sum(weights.*distances.^2);%changed from 'dissimilarities' to 'distances'
S = sqrt(sumDiffSq./sumDissSq);

if nargout > 1
    [n,p] = size(Y);
    grad = zeros(n,p);
    if sumDiffSq > 0
        if all(distances > 0)
            dS = squareform(weights.*diffs ./ distances);
            repcols = zeros(1,n);
            for i = 1:p
                repcols = repcols + 1;
                dY = Y(:,repcols) - Y(:,repcols)';
                grad(:,i) = sum(dS.*dY,2) ./ (sumDissSq.*S);
            end
        else
            error(message('stats:mdscale:ColocatedPoints'));
        end
    end
end
end

%==========================================================================

function [S,grad] = sstressCrit(Y,disparities,weights)
%SSTRESS Squared stress criterion for nonmetric multidimensional scaling.

% The Euclidean norm of the differences between the squared distances and
% the squared disparities, normalized by the Euclidean norm of the squared
% distances.
%
% Zero weights are ok as long as the corresponding dissimilarities are
% finite.  Zero distances are also OK:  this criterion is differentiable
% even when two points are coincident.

% The normalization used here is sum(distances.^4).  Some authors instead
% use sum(disparities.^4).
distances = pdist(Y);
diffs = distances.^2 - disparities.^2;
sumDiffSq = sum(weights.*diffs.^2);
sumDist4th = sum(weights.*distances.^4);
S = sqrt(sumDiffSq ./ sumDist4th);
if nargout > 1
    [n,p] = size(Y);
    grad = zeros(n,p);
    if sumDiffSq > 0
        dS = squareform(weights.* (diffs./sumDiffSq - (distances.^2)./sumDist4th));
        repcols = zeros(1,n);
        for i = 1:p
            repcols = repcols + 1;
            dY = Y(:,repcols) - Y(:,repcols)';
            grad(:,i) = sum(dS.*dY,2) .* S .* 2;
        end
    end
end
end

%==========================================================================

function [S,grad] = metricSStressCrit(Y,dissimilarities,weights)
%METRICSSTRESS Squared stress criterion for metric MDS.

% The Euclidean norm of the differences between the squared distances and
% the squared dissimilarities, normalized by the Euclidean norm of the
% squared dissimilarities.
%
% Zero weights are ok as long as the corresponding dissimilarities are
% finite.  Zero distances are also OK:  this criterion is differentiable
% even when two points are coincident.

distances = pdist(Y);
diffs = distances.^2 - dissimilarities.^2;
sumDiffSq = sum(weights.*diffs.^2);
sumDiss4th = sum(weights.*dissimilarities.^4);
S = sqrt(sumDiffSq./sumDiss4th);
if nargout > 1
    [n,p] = size(Y);
    grad = zeros(n,p);
    if sumDiffSq > 0
        dS = squareform(weights.*diffs);
        repcols = zeros(1,n);
        for i = 1:p
            repcols = repcols + 1;
            dY = Y(:,repcols) - Y(:,repcols)';
            grad(:,i) = 2 .* sum(dS.*dY,2) ./ (sumDiss4th.*S);
        end
    end
end
end

%==========================================================================

function [S,grad] = sammonCrit(Y,dissimilarities,weights)
%SAMMON Sammon mapping criterion for metric multidimensional scaling.

% The sum of the scaled, squared differences between the distances and
% the dissimilarities, normalized by the sum of the dissimilarities. 
% The squared differences are scaled by the dissimilarities before summing.
%
% Zero weights are ok as long as the corresponding dissimilarities are
% finite.  However, zero distances or dissimilarities will cause problems.

distances = pdist(Y);
diffs = distances - dissimilarities;
sumDiffSq = sum(weights.*diffs.^2 ./ dissimilarities);
sumDiss = sum(weights.*dissimilarities);
S = sumDiffSq ./ sumDiss;
if nargout > 1
    [n,p] = size(Y);
    grad = zeros(n,p);
    if sumDiffSq > 0
        if all(distances > 0)
            dS = squareform(weights.*diffs ./ (distances.*dissimilarities));
            repcols = zeros(1,n);
            for i = 1:p
                repcols = repcols + 1;
                dY = Y(:,repcols) - Y(:,repcols)';
                grad(:,i) = 2 .* sum(dS.*dY,2) ./ sumDiss;
            end
        else
            error(message('stats:mdscale:ColocatedPoints'));
        end
    end
end
end

%==========================================================================

function [S,grad] = strainCrit(Y,A,weights,obsWeights)
%STRAIN Strain criterion for metric multidimensional scaling.

% Let Y be a minimizer of norm(A-Yc*Yc'), where Yc = (I-ones(n)/n)*Y = P*Y
% (i,e,, Y centered at the origin), and A = -0.5*(D.^2).  Then Yc is also a
% minimizer of norm(B-Yc*Yc'), where B = P*A*P (the two quantities differ by
% norm(B-A), a constant).  The latter is exactly the criterion that is
% minimized in CMDS, but now we have an equivalent criterion that is suitable
% for weighting.
%
% When there are weights, we use a weighted mean to center Y.  This means that
% when all dissimilarities to a given point are zero, that point is entirely
% ignored in the calculation of strain (and it will conveniently be placed at
% the origin because its norm can be minimized separately).
%
% Zero weights are ok as long as the corresponding dissimilarities are finite.

[n,p] = size(Y);
Yc = bsxfun(@minus,Y,sum(bsxfun(@times,Y,obsWeights),1));
diffs = (A - Yc*Yc');
S = norm(sqrt(weights).*diffs,'fro'); % weight the squared diffs
if nargout > 1
    wdiffs = weights .* diffs;
    grad = zeros(n,p);
    repcols = zeros(1,n);
    for i = 1:p
        repcols = repcols + 1;
        dSrows = wdiffs .* Yc(:,repcols);
        dScols = wdiffs .* Yc(:,repcols)';
        grad(:,i) = (obsWeights.*sum(sum(dSrows+dScols,2),1) - 2.*sum(dScols,2)) ./ S;
    end
end
end

function yhat = lsqisotonic(x,y,w)
%LSQISOTONIC Isotonic least squares.
%   YHAT = LSQISOTONIC(X,Y) returns a vector of values that minimize the
%   sum of squares (Y - YHAT).^2 under the monotonicity constraint that
%   X(I) > X(J) => YHAT(I) >= YHAT(J), i.e., the values in YHAT are
%   monotonically non-decreasing with respect to X (sometimes referred
%   to as "weak monotonicity").  LSQISOTONIC uses the "pool adjacent
%   violators" algorithm.
%
%   If X(I) == X(J), then YHAT(I) may be <, ==, or > YHAT(J) (sometimes
%   referred to as the "primary approach").  If ties do occur in X, a plot
%   of YHAT vs. X may appear to be non-monotonic at those points.  In fact,
%   the above monotonicity constraint is not violated, and a reordering
%   within each group of ties, by ascending YHAT, will produce the desired
%   appearance in the plot.
%
%   YHAT = LSQISOTONIC(X,Y,W) performs weighted isotonic regression using
%   the non-negative weights in W.

%   Copyright 2003-2006 The MathWorks, Inc.


%   References:
%      [1] Kruskal, J.B. (1964) "Nonmetric multidimensional scaling: a
%          numerical method", Psychometrika 29:115-129.
%      [2] Cox, R.F. and Cox, M.A.A. (1994) Multidimensional Scaling,
%          Chapman&Hall.

n = numel(x);
if nargin<3
    yclass = superiorfloat(x,y);
else
    yclass = superiorfloat(x,y,w);
end

% Sort points ascending in x, break ties with y.
[xyord,ord] = sortrows([x(:) y(:)]); iord(ord) = 1:n;
xyord = double(xyord);

% Initialize fitted values to the given values.
yhat = xyord(:,2);

block = 1:n;
if (nargin == 3) && ~isempty(w)
    w = double(w(:)); w = w(ord); % reorder w as a column

    % Merge zero-weight points with preceding pos-weighted point (or
    % with the following pos-weighted point if at start).
    posWgts = (w > 0);
    if any(~posWgts)
        idx = cumsum(posWgts); idx(idx == 0) = 1;
        w = w(posWgts);
        yhat = yhat(posWgts);
        block = idx(block);
    end

else
    w = ones(size(yhat));
end

% Written by Maigo on 8/14/2012 to reduce the complexity from O(n^2) to O(n)
n = length(yhat);
b = 0; bstart = zeros(1,n); bend = zeros(1,n);
for i = 1:n
    b = b + 1;
    yhat(b) = yhat(i);
    w(b) = w(i);
    bstart(b) = i; bend(b) = i;
    while b > 1 && yhat(b) < yhat(b-1)
        yhat(b-1) = (yhat(b-1) * w(b-1) + yhat(b) * w(b)) / (w(b-1) + w(b));
        w(b-1) = w(b-1) + w(b);
        bend(b-1) = bend(b);
        b = b - 1;
    end
end
idx = zeros(1,n);
for i = 1:b
    idx(bstart(i) : bend(i)) = i;
end
block = idx(block);
% Maigo end

% Broadcast merged blocks out to original points, and put back in the
% original order and shape.
yhat = yhat(block);
yhat = reshape(yhat(iord), size(y));
if isequal(yclass,'single')
    yhat = single(yhat);
end
end

