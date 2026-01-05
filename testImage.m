%% PRÀCTICA VC - RECONEIXEMENT DE SENYALS DE TRÀNSIT
% Script per detectar i identificar senyals de tràfic

clear; clc; close all;

% 1. Verificar i carregar models
if ~exist('trainedModelForma.mat', 'file')
    error('No s''ha trobat trainedModelForma.mat');
end
if ~exist('trainedModelColor.mat', 'file')
    error('No s''ha trobat trainedModelColor.mat');
end
if ~exist('trainedModelCircBlanco.mat', 'file')
    error('No s''ha trobat trainedModelCircBlanco.mat');
end
if ~exist('trainedModelCircAzul.mat', 'file')
    error('No s''ha trobat trainedModelCircAzul.mat');
end

load('trainedModelForma.mat');
load('trainedModelColor.mat');
load('trainedModelCircBlanco.mat');
load('trainedModelCircAzul.mat');

% 2. Seleccionar imatge
[filename, path] = uigetfile('*.png;*.jpg;*.jpeg', 'Selecciona señal');
if filename == 0, return; end
img = imread(fullfile(path, filename));

% 3. Extraer descriptor
[desc_forma, desc_color, desc_detall] = extractDescriptors4Models(img);

% 4. Preparar datos
tabla_forma = array2table(desc_forma, 'VariableNames', {
    'Circularity', 'Eccentricity', 'Solidity', 'Extent', ...
    'AspectRatio', 'AxisRatio', 'Compactness', 'FormFactor', ...
    'NumVertices', 'Fourier1', 'Fourier2', 'Fourier3', ...
    'Perimeter', 'Area', 'Rectangularity', 'EquivDiameter'});

tabla_color = array2table(desc_color, 'VariableNames', {
    'PctRed','PctBlue','PctWhite','PctYellow','MeanRed','MeanGreen','MeanBlue',...
    'MeanHue','MeanSaturation','MeanValue','StdRed','StdBlue','RedBlueRatio',...
    'RedGreenRatio','ColorEntropy'});

% Asegurar que desc_detall tenga 30 elementos
if length(desc_detall) < 30
    desc_detall = [desc_detall, zeros(1, 30-length(desc_detall))];
end
tabla_detall = array2table(desc_detall(1:30), 'VariableNames', {
    'PctRed','PctBlue','PctWhite','PctYellow','PctBlack','MeanRed','MeanGreen',...
    'MeanBlue','StdRed','StdGreen','HueMean','HueStd','SaturationMean',...
    'SaturationStd','Circularity','Solidity','Extent','Compactness','Fourier1',...
    'Fourier2','Fourier3','Fourier4','PctEdges','Contrast','Energy','Homogeneity',...
    'SymmetryX','SymmetryY','Texton1','Texton2'});

% 5. Clasificar JERÁRQUICAMENTE
forma_pred = lower(char(trainedModelForma.predictFcn(tabla_forma)));
fprintf('Forma detectada: %s\n', forma_pred);

if contains(forma_pred, 'octagonal')
    % Si es octogonal -> es un STOP
    senal = 'STOP';
    fprintf('STOP detectado\n');
    
elseif contains(forma_pred, 'triangular')
    % Si es triangular -> es VIANANT
    senal = 'VIANANT';
    fprintf('VIANANT detectado\n');
    
elseif contains(forma_pred, 'circular')
    % Solo para señales circulares consultamos el modelo de color
    color_pred = lower(char(trainedModelColor.predictFcn(tabla_color)));
    fprintf('Color detectado: %s\n', color_pred);
    
    % Verificar porcentaje de rojo para evitar falsos positivos
    pct_red = desc_color(1);
    if pct_red > 0.7 
            senal = 'd_prohibida';
    else
        % Clasificación normal
        if contains(color_pred, 'red')
            senal = 'd_prohibida';
        elseif contains(color_pred, 'white')
            senal = upper(char(trainedModelCircBlanco.predictFcn(tabla_detall)));
        elseif contains(color_pred, 'blue')
            senal = upper(char(trainedModelCircAzul.predictFcn(tabla_detall)));
        else
            senal = 'CIRCULAR DESCONOCIDA';
        end
    end
else
    senal = 'DESCONOCIDA';
end

% 6. Ajustar nombre para visualización
switch upper(senal)
    case {'STOP', 'VIANANT'}
        nombre_mostrar = senal;
    case 'D_PROHIBIDA'
        nombre_mostrar = 'PROHIBIDO';
    otherwise
        nombre_mostrar = strrep(senal, '_', ' ');
        nombre_mostrar = upper(nombre_mostrar);
end

% 7. Mostrar resultado detallado
fprintf('\n════════════════════════════════════════\n');
fprintf(' RESULTADO FINAL: %s\n', nombre_mostrar);
fprintf('   Forma: %s\n', forma_pred);
fprintf('════════════════════════════════════════\n');

% 8. Visualizar imagen con resultado
figure, imshow(img), title(['La senyal es: ' nombre_mostrar])