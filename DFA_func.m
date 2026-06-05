%%%  https://www.mathworks.com/matlabcentral/fileexchange/67889-detrended-fluctuation-analysis-dfa
function[A,F] = DFA_func(data, pts, order, PLOT)
% -----------------------------------------------------
% DESCRIPTION:
% Function for the DFA analysis.
% INPUTS: 
% data: a one-dimensional data vector.
% pts: sizes of the windows/bins at which to evaluate the fluctuation
% order: (optional) order of the polynomial for the local trend correction.
% if not specified, order == 1;
% OUTPUTS: 
% A: a 2x1 vector. A(1) is the scaling coefficient "alpha",
% A(2) the intercept of the log-log regression, useful for plotting (see examples).
% F: A vector of size Nx1 containing the fluctuations corresponding to the
% windows specified in entries in pts.
% -----------------------------------------------------
% Checking the inputs
if nargin == 4
   order = 1; 
end
sz = size(data);
if sz(1)< sz(2)
    data = data';
end
exit = 0;
if min(pts) == order+1
        disp(['WARNING: The smallest window size is ' num2str(min(pts)) '. DFA order is ' num2str(order) '.'])
        disp('This severly affects the estimate of the scaling coefficient')
        disp('(If order == [] (so 1), the corresponding fluctuation is zero.)')
elseif min(pts) < (order+1)
        disp(['ERROR: The smallest window size is ' num2str(min(pts)) '. DFA order is ' num2str(order) ':'])
        disp(['Aborting. The smallest window size should be of ' num2str(order+1) ' points at least.'])
        exit = 1;
end
if exit == 1
    return
end
% DFA
npts = numel(pts);
F = zeros(npts,1);
N = length(data);
for h = 1:npts
    
    w = pts(h);
    
    n = floor(N/w);
    Nfloor = n*pts(h);
    D = data(1:Nfloor);
    
    y  = cumsum(D-mean(D)); %  (D-mean(D));
    
    bin = 0:w:(Nfloor-1);
    vec = 1:w;
    
    coeff = arrayfun(@(j) polyfit(vec',y(bin(j) + vec),order),1:n,'uni',0);
    y_hat = cell2mat(cellfun(@(y) polyval(y,vec),coeff,'uni',0));
    F(h)  = mean((y - y_hat').^2)^0.5;
    
end

A = polyfit(log10(pts),log10(F)',1);

if PLOT == 1
figure;
scatter(log10(pts),log10(F))
plot_fun = @(xp,A,ord) polyval(A,log10(xp));

hold on
x =  pts ;
plot(log10(x),plot_fun(x,A),'--')
xlabel('log_{10} W'), ylabel('log_{10} F(W)');
legend(['\alpha = ' num2str( sprintf('%.3f',A(1) ))],'Location','northwest');

hold off

end



end

