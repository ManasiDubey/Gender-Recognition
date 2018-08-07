clear all;


setDir  = fullfile('images');
imds = imageDatastore(setDir,'IncludeSubfolders',true,'LabelSource','foldernames');
[trainingSet,testSet] = splitEachLabel(imds,0.3,'randomize');
bag = bagOfFeatures(trainingSet);
categoryClassifier = trainImageCategoryClassifier(trainingSet,bag);
confMatrix = evaluate(categoryClassifier,testSet)
mean(diag(confMatrix))
img = imread(fullfile(setDir,'boy','201501079_goggles.jpg'));
%img = imread('ak.jpg');
[labelIdx, score] = predict(categoryClassifier,img);
gender=categoryClassifier.Labels(labelIdx)
imshow(img);
title(gender);



% faceDatabase = imageSet('images','recursive'); %load ('faces.mat')
% x1=faceDatabase(1,1).ImageLocation; x2=faceDatabase(1,2).ImageLocation; X
% = [x1 x2];
% 
% label =[]; for i=1:548
%    if(i<=396)
%        label{i} ='male';
%    else
%        label{i} ='female';
%    end
% end numberOfFiles = length(X); T = [];
% 
% for i=1:numberOfFiles
%     fileName = char(X(i)); %disp(i);
%     
%     if exist(fileName, 'file')
%         
%         I = imread(fileName);
%         
%         lbpFeatures = extractLBPFeatures(I,'CellSize',[32
%         32],'Normalization','None');
% 
%         Reshape the LBP features into a _number of neighbors_ -by-
%         _number of cells_ array to access histograms for each individual
%         cell. numNeighbors = 8; numBins =
%         numNeighbors*(numNeighbors-1)+3; lbpCellHists =
%         reshape(lbpFeatures,numBins,[]);
% 
%         Normalize each LBP cell histogram using L1 norm. lbpCellHists =
%         bsxfun(@rdivide,lbpCellHists,sum(lbpCellHists));
% 
%         Reshape the LBP features vector back to 1-by- _N_ feature
%         vector. lbpFeatures = reshape(lbpCellHists,1,[]); T = [T
%         lbpFeatures'];
% 
%     end
% end
% 
% [m, A, Eigenfaces] = EigenfaceCore(T); OutputName = Recognition(T(:,420),
% m, A, Eigenfaces); if(max(size(T(1,:)))<=396) output =
% char(faceDatabase(1,1).ImageLocation(OutputName)); else output =
% char(faceDatabase(1,2).ImageLocation(OutputName)); end
