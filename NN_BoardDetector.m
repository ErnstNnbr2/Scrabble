function Nothin = ANN_BoardDetector()
% Indentify slurry using ANN
close all;clc
IName = {'E:\WERK\ScrabbleRef\ScrabbleImages\EmptyBoard1.JPG',...
    'E:\WERK\ScrabbleRef\ScrabbleImages\Extra\IMG_0002.JPG',...
    'E:\WERK\ScrabbleRef\ScrabbleImages\Extra\IMG_9995.JPG',...
    'E:\WERK\ScrabbleRef\Label_EmptyBoard1.png',...
    'E:\WERK\ScrabbleRef\Label_IMG_0002.png',...
    'E:\WERK\ScrabbleRef\Label_IMG_9995.png'};
Feat = [];labels = [];
for i = 1:3
I = imresize(imread(IName{i}),0.1);
I = im2double(I);
L = imresize(imread(IName{i+3}),0.1);
L = imfill(L(:,:,1)>0,'holes');

% Neighbor pixel vals
R = I(:,:,1);G = I(:,:,2);B = I(:,:,3);
RP = im2col(padarray(R,[2 2],0),[5 5],'sliding');
GP = im2col(padarray(G,[2 2],0),[5 5],'sliding');
BP = im2col(padarray(B,[2 2],0),[5 5],'sliding');

% Get features of slurry and "background"
%=========================================================================
Feat = [Feat;[RP',GP',BP']];
labels = [labels;L(:)];
end
clear RP;clear GP;clear BP
clear R;clear G;clear B
randomize_feat_ind = randperm(size(Feat,1));
Feat = Feat(randomize_feat_ind,:);
labels = labels(randomize_feat_ind,:);
%==========================================================================
% Prep ANN parameters
Theta1 = RandInitPar(size(Feat,2), 50);
Theta2 = RandInitPar(50, max(labels));
%==========================================================================
% Do optimization
options = optimset('MaxIter', 10);
J = @(p) CostFunc(labels,Feat,p,size(Theta1),size(Theta2));
[nn_params, ~] = fmincg(J,[Theta1(:)',Theta2(:)']',options);
%==========================================================================
% Check prediction

Theta1 = reshape(nn_params(1:numel(Theta1)),size(Theta1));
Theta2 = reshape(nn_params(numel(Theta1)+1:end),size(Theta2));
names = {'EmptyBoard5','FirstWord1','FirstWord2',...
    'FirstWord3','FirstWord4','FourthWord1'};
for i = 1:6
I = imresize(imread(['E:\WERK\ScrabbleRef\ScrabbleImages\',...
    names{i},'.JPG']),0.1);
I = im2double(I);
R = I(:,:,1);G = I(:,:,2);B = I(:,:,3);
RP = im2col(padarray(R,[2 2],0),[5 5],'sliding');
GP = im2col(padarray(G,[2 2],0),[5 5],'sliding');
BP = im2col(padarray(B,[2 2],0),[5 5],'sliding');
Feat = [RP',GP',BP'];

m = size(Feat,1);
a1 = [ones(m,1),Feat];
z2 = a1*Theta1';
a2 = [ones(size(a1,1),1),sigmoid(z2)];
h = sigmoid(a2*Theta2');

lab1 = reshape(h,size(L));
figure,imshow(lab1,[])
pause
end
Nothin = nn_params;
end

function [J,Grad] = CostFunc(Y,X,param,sizeT1,sizeT2)
% Compute cost and gradient for ANN
Theta1 = reshape(param(1:sizeT1(1)*sizeT1(2)),sizeT1);
Theta2 = reshape(param(sizeT1(1)*sizeT1(2)+1:end),sizeT2);
m = size(X,1);

a1 = [ones(m,1),X];
z2 = a1*Theta1';
a2 = [ones(size(a1,1),1),sigmoid(z2)];
h = sigmoid(a2*Theta2');

j = -Y.*log(h) - (1-Y).*log(1-h); %cost
J = sum(j(:))/m;

d3 = h - Y;
d2 = d3*Theta2.*sigmoidGradient([ones(size(z2,1),1),z2]); % grad
d2 = d2(:,2:end);

D1 = d2'*a1;
D2 = d3'*a2;

Theta1_grad = D1./ m;
Theta2_grad = D2./ m;
Grad = [Theta1_grad(:);Theta2_grad(:)];
% =========================================================================
end

function g = sigmoid(z)
% SIGMOID
% =========================================================================
g = 1.0 ./ (1.0 + exp(-z));
% =========================================================================
end

function g = sigmoidGradient(z)
% SIGMOIDGRADIENT
% =========================================================================
g = sigmoid(z).*(1 - sigmoid(z));
% =========================================================================
end

function W = RandInitPar(L_in, L_out)
% RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
% incoming connections and L_out outgoing connections
%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights
%   of a layer with L_in incoming connections and L_out outgoing
%   connections.
%
%   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
%   the column row of W handles the "bias" terms
% =========================================================================

W = zeros(L_out, 1 + L_in);
epsilon_init = 0.12;
W = rand(L_out, 1 + L_in)*2*epsilon_init - epsilon_init;
% =========================================================================
end


