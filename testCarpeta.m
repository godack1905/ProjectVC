%% PRÀCTICA VC - RECONEIXEMENT DE SENYALS DE TRÀNSIT
% Testing amb totes les imatges d'una carpeta
clear; clc; close all;

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

% Carregar el model entrenat
load('trainedModel.mat');

test_folder = 'imatges_senyals\test\vianant';

% Obtenir totes les imatges (PNG, JPG, JPEG)
png_files = dir(fullfile(test_folder, '*.png'));
jpg_files = dir(fullfile(test_folder, '*.jpg'));
jpeg_files = dir(fullfile(test_folder, '*.jpeg'));
image_files = [png_files; jpg_files; jpeg_files];

fprintf('S''han trobat %d imatges a la carpeta: %s\n\n', length(image_files), test_folder);


%procesar imatges
clases_predichas = cell(length(image_files), 1);
for i = 1:length(image_files)
    % Llegir la imatge
    img_path = fullfile(test_folder, image_files(i).name);
    img_test = imread(img_path);
    
    % Extreure descriptors
    desc_test = extractDescriptors(img_test, true);
    
    % Crear taula amb els descriptors
    tabla_test = array2table(desc_test, 'VariableNames', descriptor_names);
    
    % Fer la prediccio
    classe = trainedModel.predictFcn(tabla_test);
    clases_predichas{i} = char(classe);
    
    % Mostrar resultat en consola
    fprintf('Imatge %d/%d: %s -> Classe: %s\n', ...
        i, length(image_files), image_files(i).name, char(classe));
end

%% RESUM DE PREDICCIONS
fprintf('\nRESUM DE PREDICCIONS\n');

% Contar quantes prediccions de cada classe
clases_unicas = unique(clases_predichas);
fprintf('\nTotal d''imatges processades: %d\n\n', length(image_files));

for i = 1:length(clases_unicas)
    clase_actual = clases_unicas{i};
    cantidad = sum(strcmp(clases_predichas, clase_actual));
    porcentaje = (cantidad / length(image_files)) * 100;
    fprintf('  %s: %d imatges (%.1f%%)\n', clase_actual, cantidad, porcentaje);
end
