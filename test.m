%% DETECTOR DE SEÑALES - VERSIÓN MÍNIMA
clear; clc; close all;

% 1. Cargar modelos
load('trainedModelForma.mat');
load('trainedModelColor.mat');
load('trainedModelCircBlanco.mat');
load('trainedModelCircAzul.mat');

% 2. Seleccionar imagen
[filename, path] = uigetfile('*.png;*.jpg;*.jpeg', 'Selecciona señal');
if filename == 0, return; end
img = imread(fullfile(path, filename));

% 3. Extraer características
[desc_forma, desc_color, desc_detall] = extractDescriptors5Models(img);

% 4. Preparar datos
tabla_forma = array2table(desc_forma, 'VariableNames', {
    'Circularity','Eccentricity','Solidity','Extent','AspectRatio','AxisRatio',...
    'Compactness','FormFactor','NumVertices','Fourier1','Fourier2','Fourier3',...
    'Perimeter','Area','Rectangularity','EquivDiameter'});

tabla_color = array2table(desc_color, 'VariableNames', {
    'PctRed','PctBlue','PctWhite','PctYellow','MeanRed','MeanGreen','MeanBlue',...
    'MeanHue','MeanSaturation','MeanValue','StdRed','StdBlue','RedBlueRatio',...
    'RedGreenRatio','ColorEntropy'});

if length(desc_detall) < 30
    desc_detall = [desc_detall, zeros(1, 30-length(desc_detall))];
end
tabla_detall = array2table(desc_detall(1:30), 'VariableNames', {
    'PctRed','PctBlue','PctWhite','PctYellow','PctBlack','MeanRed','MeanGreen',...
    'MeanBlue','StdRed','StdGreen','HueMean','HueStd','SaturationMean',...
    'SaturationStd','Circularity','Solidity','Extent','Compactness','Fourier1',...
    'Fourier2','Fourier3','Fourier4','PctEdges','Contrast','Energy','Homogeneity',...
    'SymmetryX','SymmetryY','Texton1','Texton2'});

% 5. Clasificar
forma = lower(char(trainedModelForma.predictFcn(tabla_forma)));

if contains(forma, 'octagonal')
    senal = 'STOP';
elseif contains(forma, 'triangular')
    senal = 'VIANANT';
elseif contains(forma, 'circular')
    color = lower(char(trainedModelColor.predictFcn(tabla_color)));
    
    if contains(color, 'red')
        senal = 'PROHIBIDO';
    elseif contains(color, 'white')
        senal = upper(char(trainedModelCircBlanco.predictFcn(tabla_detall)));
    elseif contains(color, 'blue')
        senal = upper(char(trainedModelCircAzul.predictFcn(tabla_detall)));
    else
        senal = 'CIRCULAR DESCONOCIDA';
    end
else
    senal = 'DESCONOCIDA';
end

% 6. Mostrar resultado en imagen
fprintf('\n✅ Señal detectada: %s\n', senal);

figure, imshow(img), title(senal);