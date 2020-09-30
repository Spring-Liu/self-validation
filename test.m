clc;clear;
addpath(genpath('.\'));
load('Tucker_syn.mat');

mr = 0.5; %missing ratio
S = size(Tucker_syn);
W = gen_W(S,mr); %generate the observation

score = selfvalidation(W,S,Tucker_syn);
