%% PRÀCTICA VC - RECONEIXEMENT DE SENYALS DE TRÀNSIT
% Testing amb imatges

% Noms dels descriptors
descriptor_names = {
    'PctRed', 'PctBlue', 'PctYellow', ...
    'MeanRed', 'MeanGreen', 'MeanBlue', ...
    'Area', 'Perimeter', 'Eccentricity', ...
    'Solidity', 'Extent', 'Circularity', ...
    'AspectRatio', 'AxisRatio', 'Compactness', ...
    'Fourier1', 'Fourier2', 'Fourier3', 'Fourier4', 'Fourier5', ...
    'PctEdges', 'Contrast', 'Correlation', ...
    'Energy', 'Homogeneity'
};

% Cargar el model entrenat
load('trainedModel.mat');

% Imatge
img_test = imread('imatges_senyals\test\vianant\035_1_0004.png');
desc_test = extractDescriptors(img_test);

% Obtenir els descriptors de la imatge
tabla_test = array2table(desc_test, 'VariableNames', descriptor_names);

% Reconeixer la imatge
classe = trainedModel.predictFcn(tabla_test);

figure, imshow(img_test), title(['La senyal es: ' char(classe)])
