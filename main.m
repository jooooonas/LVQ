close all

load lvqdata.mat

% extract P and N
P = length(lvqdata(:,1));
N = length(lvqdata(1,:));

% create labels: 1st half of data points is label 1, 2nd half is label 2
labels = ones(P,1);
labels(P/2+1:P,1) = 2;

K = 1; % set number of prototypes per class
n = 0.002; % set learning rate
t_max = 150; % set max number of epochs / iterations

% execute algorithm
lvq1(P, N, lvqdata, labels, K, n, t_max);
K = 2;
lvq1(P, N, lvqdata, labels, K, n, t_max);

% bonus task:
% K = 3;
% lvq1(P, N, lvqdata, labels, K, n, t_max);
% K = 4;
% lvq1(P, N, lvqdata, labels, K, n, t_max);