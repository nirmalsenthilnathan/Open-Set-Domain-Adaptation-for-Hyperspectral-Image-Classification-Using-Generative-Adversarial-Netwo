clc;
clear all;
close all;

%Load and prepare data:
load('SalinasA.mat');
load('SalinasA_gt.mat');
data= im1;
gt = salinasA_gt;

%Assign train-test ratio and calling function file:
% t=input('enter test split=');
[k k_train k_test]  = Group_train_test_split(data, gt, t);
k=k';
csvwrite('SalinasA.csv',k)

% k = Group_train_test_split(data, gt, t);
% [~, k_train] = Group_train_test_split(data, gt, t);
% [~, ~, k_test] = Group_train_test_split(data, gt, t);