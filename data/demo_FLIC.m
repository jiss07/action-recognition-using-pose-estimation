%% takes 2.5 seconds to load 21K element annotation struct
load examples.mat
imgdir = './images/';
%% display a random image
i = randi(length(examples));

img = imread([imgdir,'/',examples(i).filepath]);
cla, imagesc(img), axis image, hold on

% display torso detected by berkeley poselets
plotbox(examples(i).torsobox,'w--')

% display all the labeled joints; median of 5 annotations by mechanical turk
myplot(examples(i).coords(:,lookupPart('lsho','lelb','lwri')),'go-','linewidth',3)
myplot(examples(i).coords(:,lookupPart('rsho','relb','rwri')),'mo-','linewidth',3)
myplot(examples(i).coords(:,lookupPart('rhip','lhip')),'bo-','linewidth',3)
myplot(examples(i).coords(:,lookupPart('leye','reye','nose','leye')),'c.-','linewidth',3)

%% notes on is* fields in examples struct:
%{
istrain - used for training/cross validation
istest - used for testing
isbad - not used, because at least one part is either mislabeled, occluded, very non-frontal or too tiny
isunchecked - not used, because we didn't check whether it 'isbad' or not (simply ran out of time)

We include the 'isbad' and 'isunchecked' for future research... handling occlusions, handling outlier data in training, etc.
%}
