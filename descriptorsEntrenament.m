%% PRÀCTICA VC - EXTRACCIÓ DE DESCRIPTORS PER 5 MODELS JERÀRQUICS
% Genera taules per entrenar 5 models al Classification Learner

clear; clc; close all;

%% CONFIGURACIÓ
train_path = 'imatges_senyals/train';
categorias = {
    'd_obligatoria',        % circular, azul, flecha blanca
    'd_prohibida',          % circular, roja con franja blanca  
    'limit',                % circular, blanca y roja, numeros negros
    'no_aparcar',           % circular, azul con rayas rojas
    'no_girar',             % circular, blanca con flecha negra tachada de rojo
    'no_soroll',            % circular, blanca con bocina negra tachada de rojo
    'stop',                 % octogonal, roja con letras blancas STOP
    'vianant',              % triangular, amarilla con persona negra
    'zona_bici',            % circular, azul con bici blanca
    'zona_cotxe'            % circular, azul con coche blanco
};

%% VERIFICAR ESTRUCTURA
if ~exist(train_path, 'dir')
    fprintf('ERROR: No existeix el directori: %s\n', train_path);
    return;
end

%% INICIALITZAR ESTRUCTURES
% Model 1: Formes (triangular/circular/octogonal)
descriptors_forma = [];
labels_forma = {};

% Model 2: Color principal (roig/blau/blanc)
descriptors_color = [];
labels_color = {};

% Model 3: Circulars roges (d_prohibida)
descriptors_circ_rojo = [];
labels_circ_rojo = {};

% Model 4: Circulars blaves (d_obligatoria, no_aparcar, zona_bici, zona_cotxe)
descriptors_circ_azul = [];
labels_circ_azul = {};

% Model 5: Circulars blanques (limit, no_girar, no_soroll)
descriptors_circ_blanco = [];
labels_circ_blanco = {};

%% DESCRIPTORS PER CADA MODEL
% Noms dels descriptors per cada model
descriptor_names_forma = {
    'Circularity', 'Eccentricity', 'Solidity', 'Extent', ...
    'AspectRatio', 'AxisRatio', 'Compactness', 'FormFactor', ...
    'NumVertices', 'Fourier1', 'Fourier2', 'Fourier3', ...
    'Perimeter', 'Area', 'Rectangularity', 'EquivDiameter'
};

descriptor_names_color = {
    'PctRed', 'PctBlue', 'PctWhite', 'PctYellow', ...
    'MeanRed', 'MeanGreen', 'MeanBlue', 'MeanHue', ...
    'MeanSaturation', 'MeanValue', 'StdRed', 'StdBlue', ...
    'RedBlueRatio', 'RedGreenRatio', 'ColorEntropy'
};

descriptor_names_detall = {
    'PctRed', 'PctBlue', 'PctWhite', 'PctYellow', 'PctBlack', ...
    'MeanRed', 'MeanGreen', 'MeanBlue', 'StdRed', 'StdGreen', ...
    'HueMean', 'HueStd', 'SaturationMean', 'SaturationStd', ...
    'Circularity', 'Solidity', 'Extent', 'Compactness', ...
    'Fourier1', 'Fourier2', 'Fourier3', 'Fourier4', ...
    'PctEdges', 'Contrast', 'Energy', 'Homogeneity', ...
    'SymmetryX', 'SymmetryY', 'Texton1', 'Texton2'
};

%% PROCÉS PER CATEGORIA
fprintf('=== EXTRACCIÓ DE DESCRIPTORS ===\n');

for cat_idx = 1:length(categorias)
    categoria = categorias{cat_idx};
    carpeta = fullfile(train_path, categoria);
    
    if ~exist(carpeta, 'dir')
        fprintf('ADVERTÈNCIA: No existeix la carpeta: %s\n', carpeta);
        continue;
    end
    
    fprintf('Processant: %s\n', categoria);
    
    % Llistar imatges
    archivos = dir(fullfile(carpeta, '*.png'));
    if isempty(archivos)
        archivos = dir(fullfile(carpeta, '*.jpg'));
    end
    if isempty(archivos)
        archivos = dir(fullfile(carpeta, '*.jpeg'));
    end
    
    if isempty(archivos)
        fprintf('  No s''han trobat imatges\n');
        continue;
    end
    
    for img_idx = 1:length(archivos)
        img_path = fullfile(carpeta, archivos(img_idx).name);
        
        try
            img = imread(img_path);
            
            if isempty(img) || ndims(img) < 3
                continue;
            end
            
            % Extreure descriptors jeràrquics
            [desc_forma, desc_color, desc_detall] = extractDescriptors5Models(img);
            
            % Determinar etiquetes segons categoria
            % Etiqueta per Model 1 (Forma)
            if strcmp(categoria, 'stop')
                label_forma = 'octagonal';
            elseif strcmp(categoria, 'vianant')
                label_forma = 'triangular';
            else
                label_forma = 'circular';
            end
            
            % Etiqueta per Model 2 (Color)
            if ismember(categoria, {'d_prohibida', 'stop'})
                label_color = 'red';
            elseif ismember(categoria, {'d_obligatoria', 'no_aparcar', 'zona_bici', 'zona_cotxe'})
                label_color = 'blue';
            else
                label_color = 'white';
            end
            
            % Afegir a Model 1 (Formes)
            if ~isempty(desc_forma)
                descriptors_forma = [descriptors_forma; desc_forma];
                labels_forma = [labels_forma; label_forma];
            end
            
            % Afegir a Model 2 (Colors)
            if ~isempty(desc_color)
                descriptors_color = [descriptors_color; desc_color];
                labels_color = [labels_color; label_color];
            end
            
            % Afegir a Models 3-5 segons tipus
            if ~isempty(desc_detall)
                % Model 3: Circulars roges
                if strcmp(label_color, 'red') && strcmp(label_forma, 'circular')
                    descriptors_circ_rojo = [descriptors_circ_rojo; desc_detall];
                    labels_circ_rojo = [labels_circ_rojo; categoria];
                    
                % Model 4: Circulars blaves
                elseif strcmp(label_color, 'blue') && strcmp(label_forma, 'circular')
                    descriptors_circ_azul = [descriptors_circ_azul; desc_detall];
                    labels_circ_azul = [labels_circ_azul; categoria];
                    
                % Model 5: Circulars blanques
                elseif strcmp(label_color, 'white') && strcmp(label_forma, 'circular')
                    descriptors_circ_blanco = [descriptors_circ_blanco; desc_detall];
                    labels_circ_blanco = [labels_circ_blanco; categoria];
                end
            end
            
            if mod(img_idx, 20) == 0
                fprintf('  Processades %d/%d\n', img_idx, length(archivos));
            end
            
        catch ME
            fprintf('  Error en %s\n', archivos(img_idx).name);
        end
    end
end

%% CREAR I GUARDAR TAULES
fprintf('\n=== CREANT TAULES PER CLASSIFICATION LEARNER ===\n');

% Model 1: Formes
if ~isempty(descriptors_forma)
    train_table_forma = array2table(descriptors_forma, 'VariableNames', descriptor_names_forma);
    train_table_forma.Class = categorical(labels_forma);
    save('data_model1_forma.mat', 'train_table_forma');
    fprintf('Model 1 (Formes): %d mostres, %d classes\n', ...
        size(descriptors_forma, 1), length(unique(labels_forma)));
end

% Model 2: Colors
if ~isempty(descriptors_color)
    train_table_color = array2table(descriptors_color, 'VariableNames', descriptor_names_color);
    train_table_color.Class = categorical(labels_color);
    save('data_model2_color.mat', 'train_table_color');
    fprintf('Model 2 (Colors): %d mostres, %d classes\n', ...
        size(descriptors_color, 1), length(unique(labels_color)));
end

% Model 3: Circulars roges
if ~isempty(descriptors_circ_rojo)
    train_table_circ_rojo = array2table(descriptors_circ_rojo, 'VariableNames', descriptor_names_detall);
    train_table_circ_rojo.Class = categorical(labels_circ_rojo);
    save('data_model3_circ_rojo.mat', 'train_table_circ_rojo');
    fprintf('Model 3 (Circulars roges): %d mostres, %d classes\n', ...
        size(descriptors_circ_rojo, 1), length(unique(labels_circ_rojo)));
end

% Model 4: Circulars blaves
if ~isempty(descriptors_circ_azul)
    train_table_circ_azul = array2table(descriptors_circ_azul, 'VariableNames', descriptor_names_detall);
    train_table_circ_azul.Class = categorical(labels_circ_azul);
    save('data_model4_circ_azul.mat', 'train_table_circ_azul');
    fprintf('Model 4 (Circulars blaves): %d mostres, %d classes\n', ...
        size(descriptors_circ_azul, 1), length(unique(labels_circ_azul)));
end

% Model 5: Circulars blanques
if ~isempty(descriptors_circ_blanco)
    train_table_circ_blanco = array2table(descriptors_circ_blanco, 'VariableNames', descriptor_names_detall);
    train_table_circ_blanco.Class = categorical(labels_circ_blanco);
    save('data_model5_circ_blanco.mat', 'train_table_circ_blanco');
    fprintf('Model 5 (Circulars blanques): %d mostres, %d classes\n', ...
        size(descriptors_circ_blanco, 1), length(unique(labels_circ_blanco)));
end

fprintf('\n✔ Taules guardades. Ara pots:\n');
fprintf('1. Obrir Classification Learner\n');
fprintf('2. Importar cada data_modelX.mat\n');
fprintf('3. Entrenar 5 models separats\n');