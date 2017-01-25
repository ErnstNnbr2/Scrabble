function output = TestScrabble()
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
load('parameters.mat')
Name = {'ScrabbleImages\',...
    'EmptyBoard1.JPG','EmptyBoard2.JPG','EmptyBoard3.JPG',...
    'EmptyBoard4.JPG','EmptyBoard5.JPG','FirstWord1.JPG',...
    'FirstWord2.JPG','FirstWord3.JPG','FirstWord4.JPG',...
    'FourthWord1.JPG','FourthWord2.JPG','FourthWord3.JPG',...
    'SecondWord1.JPG','SecondWord2.JPG','ThirdWord1.JPG',...
    'ThirdWord2.JPG'};
for i = 2:17
    I = imresize(imread([Name{1},Name{i}]),0.2);
    [L,J] = TestOnIm(I,par_letter,par_board);
    
    subplottight(1,2,1),imshow(L,[])
    subplottight(1,2,2),imshow(J,[])
    pause
end
end

function [out1,out2] = TestOnIm(I,parl,parb)
Theta1 = reshape(parb(1:(50*76)),[50,76]);
Theta2 = reshape(parb((50*76)+1:end),[1,51]);
I = im2double(I);
R = I(:,:,1);G = I(:,:,2);B = I(:,:,3);
RP = im2col(padarray(R,[2 2],0),[5 5],'sliding');
GP = im2col(padarray(G,[2 2],0),[5 5],'sliding');clear G
BP = im2col(padarray(B,[2 2],0),[5 5],'sliding');clear B
Feat = [RP',GP',BP'];
clear RP;clear GP;clear BP
m = size(Feat,1);
a1 = [ones(m,1),Feat];
z2 = a1*Theta1';
a2 = [ones(size(a1,1),1),sigmoid(z2)];
h = sigmoid(a2*Theta2');

L = bwlabel(imfill(reshape(h,size(R))>0.5,'holes'));
A = regionprops(L,'Area');[~,ind] = max([A.Area]);
L = 1*(L == ind);
L = imerode(imfill(imclose(...
    L,strel('disk',5)),'holes'),strel('disk',5));
L = cat(3,L.*I(:,:,1),L.*I(:,:,2),L.*I(:,:,3));
L = decorrstretch(L);
out1 = L;
% R = L(:,:,1);G = L(:,:,2);B = L(:,:,3);
% [idx,c] = kmeans([R(:),G(:),B(:)],5);
J = imclose((L(:,:,1)>0.75)&(L(:,:,2)>0.3)&(L(:,:,3)>0.4),...
    strel('disk',5));J = bwlabel(J);
A = regionprops(J,'Area');[~,ind] = max([A.Area]);
J = 1*(J == ind);
J = imdilate(J,strel('disk',3));
J = (1 - J.*I(:,:,1))*255;%J = uint8(J > 0);
% ocrResults = ocr(J,'TextLayout','Block');
out2 = J;
% loc = ocrResults.CharacterConfidences;
% out2 = insertObjectAnnotation(J, 'rectangle',...
%     ocrResults.CharacterBoundingBoxes(~isnan(loc),:),...
%     loc(~isnan(loc)));
end

function g = sigmoid(z)
% SIGMOID
% =========================================================================
g = 1.0 ./ (1.0 + exp(-z));
% =========================================================================
end


function h = subplottight(n,m,i)
[c,r] = ind2sub([m,n],i);
ax = subplot('Position',[(c-1)/m,1-(r/n),1/m,1/n]);
if(nargout>0)
    h = ax;
end
end
