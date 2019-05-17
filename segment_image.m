%segment_image.m 
%   Applies semantic image segmentation on the desired image. It
%   uses pre-trained network based on DeepLabv3+. The network is used via a
%   Python script to handle segmentation. Python and necessary libraries 
%   must be installed before running this script. 
%   Please read the manual before.
%   Ali Emre Kavur 17/05/2019


%% Parameters
img_name='Image1.jpg'; % Define image name here

%% Segment image via DeepLabv3+ and save to seg_map.mat file 
disp('Calling Python script...')
[status,cmdout] = system(['python segmenter.py "' img_name '"']); % Call segmenter.py from Matlab

%% Import segmentatio results to Matlab
if ~isempty(strfind(cmdout,'Successful')) && ~status % If segmenter.py is successful go on.
    load('seg_map.mat') % Load output of segmenter.py for your input
    disp('Segmentation map imported.')

    object_list = [{'background'}, {'aeroplane'}, {'bicycle'}, {'bird'}, {'boat'}, {'bottle'}, {'bus'},...
        {'car'}, {'cat'}, {'chair'}, {'cow'}, {'diningtable'}, {'dog'}, {'horse'}, {'motorbike'},...
        {'person'}, {'pottedplant'}, {'sheep'}, {'sofa'}, {'train'}, {'tv'}]; % Define object list
    C =unique(seg_map); % Determie labels inside the segmented image
    %% Ilustration
    I=imread(img_name); % Import original image to Matlab
    map=autumn(21); % Colormap for visualition of segmentation label(s)
    map(1,:)=1; % Make backgroun "1"
    colors=ind2rgb(seg_map, map); % Illustrate segmentation map
    I_seg_rgb=uint8(double(I).*colors); % Show segmentation map on the original image
    
    %% Plots
    figure;imshow(I) % Original image
    figure;imshow(I_seg_rgb) % Illustration of segmentation
    title(['Segmented object: ' object_list{C(2)+1}]) % Name of the deteced object(s)
    autoArrangeFigures(1,2,1); % Auto arrange figures --Optional
else
    disp('Error: Segmentation from Pyhon call is not successful!')
end
