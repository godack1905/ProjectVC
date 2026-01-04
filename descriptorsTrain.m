%% PRÀCTICA VC - RECONEIXEMENT DE SENYALS DE TRÀNSIT
% Script per generar les taules per entrenar 4 models amb
% el Classification Learner

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

% Model 3: Circulars blaves (d_obligatoria, no_aparcar, zona_bici, zona_cotxe)
descriptors_circ_azul = [];
labels_circ_azul = {};

% Model 4: Circulars blanques (limit, no_girar, no_soroll)
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
            [desc_forma, desc_color, desc_detall] = extractDescriptors4Models(img);
            
            if length(desc_forma) ~= 16
                fprintf('  Advertencia: desc_forma tamaño incorrecto en %s (%d)\n', ...
                    archivos(img_idx).name, length(desc_forma));
                continue; 
            end
            
            if length(desc_color) ~= 15
                fprintf('  Advertencia: desc_color tamaño incorrecto en %s (%d)\n', ...
                    archivos(img_idx).name, length(desc_color));
                continue;
            end
            
            if length(desc_detall) ~= 30
                fprintf('  Advertencia: desc_detall tamaño incorrecto en %s (%d)\n', ...
                    archivos(img_idx).name, length(desc_detall));
                continue;
            end
            
            label_forma = '';
            label_color = '';
            skip_color_model = false;
            
            % Determinar etiquetes segons categoria
            % Etiqueta per Model 1 (Forma)
            if strcmp(categoria, 'stop')
                label_forma = 'octagonal';
                skip_color_model = true;  % Excluir del modelo de color
                
            elseif strcmp(categoria, 'vianant')
                label_forma = 'triangular';
                skip_color_model = true;  % Excluir del modelo de color
                
            else
                label_forma = 'circular';
            end
            
            % Etiqueta per Model 2 (Color) - Nomes senyals circulars
            if ~skip_color_model
                if ismember(categoria, {'d_prohibida'})
                    label_color = 'red';
                elseif ismember(categoria, {'d_obligatoria', 'no_aparcar', 'zona_bici', 'zona_cotxe'})
                    label_color = 'blue';
                else
                    label_color = 'white';
                end
            end
            
            % Afegir a Model 1 (Formes) - Totes las señales
            if ~isempty(desc_forma)
                descriptors_forma = [descriptors_forma; desc_forma];
                labels_forma = [labels_forma; label_forma];
            end
            
            % Afegir a Model 2 (Colors) - Excluir stop i vianant
            if ~isempty(desc_color) && ~skip_color_model
                descriptors_color = [descriptors_color; desc_color];
                labels_color = [labels_color; label_color];
            end
            
            % Afegir a Models 3-4 segons tipus (nomes circulares)
            if ~isempty(desc_detall) && ~skip_color_model
                % Model 3: Circulars blaves
                if strcmp(label_color, 'blue') && strcmp(label_forma, 'circular')
                    descriptors_circ_azul = [descriptors_circ_azul; desc_detall];
                    labels_circ_azul = [labels_circ_azul; categoria];
                    
                % Model 4: Circulars blanques
                elseif strcmp(label_color, 'white') && strcmp(label_forma, 'circular')
                    descriptors_circ_blanco = [descriptors_circ_blanco; desc_detall];
                    labels_circ_blanco = [labels_circ_blanco; categoria];
                end
            end
            
            if mod(img_idx, 20) == 0
                fprintf('  Processades %d/%d\n', img_idx, length(archivos));
            end
            
        catch ME
            fprintf('  Error en %s: %s\n', archivos(img_idx).name, ME.message);
        end
    end
end

%% Crear i guardar les taules
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

% Model 3: Circulars blaves
if ~isempty(descriptors_circ_azul)
    train_table_circ_azul = array2table(descriptors_circ_azul, 'VariableNames', descriptor_names_detall);
    train_table_circ_azul.Class = categorical(labels_circ_azul);
    save('data_model3_circ_azul.mat', 'train_table_circ_azul');
    fprintf('Model 3 (Circulars blaves): %d mostres, %d classes\n', ...
        size(descriptors_circ_azul, 1), length(unique(labels_circ_azul)));
end

% Model 4: Circulars blanques
if ~isempty(descriptors_circ_blanco)
    train_table_circ_blanco = array2table(descriptors_circ_blanco, 'VariableNames', descriptor_names_detall);
    train_table_circ_blanco.Class = categorical(labels_circ_blanco);
    save('data_model4_circ_blanco.mat', 'train_table_circ_blanco');
    fprintf('Model 4 (Circulars blanques): %d mostres, %d classes\n', ...
        size(descriptors_circ_blanco, 1), length(unique(labels_circ_blanco)));
end