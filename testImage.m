%% PRÀCTICA VC - RECONEIXEMENT DE SENYALS DE TRÀNSIT
% Script per detectar i identificar senyals de tràfic


clear; clc; close all;

% 1. Carregar models
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
    'Circularity','Eccentricity','Solidity','Extent','AspectRatio','AxisRatio',...
    'Compactness','FormFactor','NumVertices','Fourier1','Fourier2','Fourier3',...
    'Fourier4','Fourier5','Rectangularity','EquivDiameter','Octagonality',...
    'Angularity','RotationSymmetry','Convexity'});

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
% Primero: clasificar FORMA (con nuevo descriptor Octagonality)
forma_pred = lower(char(trainedModelForma.predictFcn(tabla_forma)));
fprintf('Forma detectada: %s\n', forma_pred);

% Mostrar valor de octogonalidad para debugging
octagonality_value = desc_forma(17);
fprintf('Valor de Octagonality: %.3f\n', octagonality_value);

if contains(forma_pred, 'octagonal')
    % Si es octogonal -> es un STOP
    senal = 'STOP';
    fprintf('✓ STOP detectado (Octagonality: %.3f)\n', octagonality_value);
    
elseif contains(forma_pred, 'triangular')
    % Si es triangular -> es VIANANT
    senal = 'VIANANT';
    fprintf('✓ VIANANT detectado\n');
    
elseif contains(forma_pred, 'circular')
    % Solo para señales circulares consultamos el modelo de color
    color_pred = lower(char(trainedModelColor.predictFcn(tabla_color)));
    fprintf('Color detectado: %s\n', color_pred);
    
    % Verificar porcentaje de rojo para evitar falsos positivos
    pct_red = desc_color(1);
    if pct_red > 0.7 && octagonality_value > 0.4
        % Si tiene mucho rojo Y alta octogonalidad, podría ser STOP mal clasificado
        fprintf('⚠ ADVERTENCIA: Mucho rojo (%.1f%%) con octogonalidad (%.3f)\n', ...
            pct_red*100, octagonality_value);
        fprintf('  Revisando...\n');
        
        % Verificar si podría ser STOP
        if octagonality_value > 0.6
            senal = 'STOP';
            fprintf('  ✓ Corregido a STOP (alta octogonalidad)\n');
        else
            senal = 'd_prohibida';
        end
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
fprintf('✅ RESULTADO FINAL: %s\n', nombre_mostrar);
fprintf('   Forma: %s\n', forma_pred);
fprintf('   Octagonality: %.3f\n', octagonality_value);
fprintf('════════════════════════════════════════\n');

% 8. Visualizar
figure('Position', [100, 100, 1000, 500]);

% Imagen original
subplot(1,3,1);
imshow(img);
title('Imagen Original', 'FontSize', 12);

% Gráfico de características
subplot(1,3,2);
bar_data = [desc_forma(1), desc_forma(17), desc_color(1)]; % Circularidad, Octogonalidad, %Rojo
bar(bar_data);
set(gca, 'XTickLabel', {'Circularidad', 'Octogonalidad', '% Rojo'});
title('Características Principales', 'FontSize', 12);
ylabel('Valor');
grid on;
ylim([0 1.2]);

% Resultado final
subplot(1,3,3);
imshow(img);
title(['SEÑAL: ' nombre_mostrar], 'FontSize', 14, 'FontWeight', 'bold', 'Color', [0, 0.5, 0]);

% Información adicional
info_text = sprintf('Forma: %s\nOctogonalidad: %.2f\n%% Rojo: %.1f%%', ...
    forma_pred, octagonality_value, desc_color(1)*100);
annotation('textbox', [0.02, 0.02, 0.96, 0.08], ...
    'String', info_text, ...
    'FontSize', 9, 'EdgeColor', 'none', ...
    'BackgroundColor', [0.95, 0.95, 0.95], ...
    'HorizontalAlignment', 'center');