%% PRÀCTICA VC - RECONEIXEMENT DE SENYALS DE TRÀNSIT
% Script per generar les taules per entrenar 4 models amb
% el Classification Learner
% VERSIÓ FUSIONADA: Estructura 4 models + Gestió d'errors robusta

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

%% VERIFICAR ESTRUCTURA DE CARPETES
fprintf('=== VERIFICANT ESTRUCTURA DE CARPETES ===\n');
fprintf('Ruta base: %s\n', train_path);

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

%% INICIALITZAR COMPTADORS D'ERRORS
total_processed = 0;
total_errors = 0;
total_discarded = 0;

%% PROCÉS PER CATEGORIA
fprintf('\n=== EXTRACCIÓ DE DESCRIPTORS ===\n');

for cat_idx = 1:length(categorias)
    categoria = categorias{cat_idx};
    carpeta = fullfile(train_path, categoria);
    
    if ~exist(carpeta, 'dir')
        fprintf('ADVERTÈNCIA: No existeix la carpeta: %s\n', carpeta);
        continue;
    end
    
    fprintf('Processant categoria: %s\n', categoria);
    
    % Llistar imatges (buscar tots els formats)
    archivos_png = dir(fullfile(carpeta, '*.png'));
    archivos_jpg = dir(fullfile(carpeta, '*.jpg'));
    archivos_jpeg = dir(fullfile(carpeta, '*.jpeg'));
    archivos = [archivos_png; archivos_jpg; archivos_jpeg];
    
    if isempty(archivos)
        fprintf('  No s''han trobat imatges en %s\n', carpeta);
        continue;
    end
    
    fprintf('  Trobades %d imatges\n', length(archivos));
    processed_in_cat = 0;
    
    for img_idx = 1:length(archivos)
        img_path = fullfile(carpeta, archivos(img_idx).name);
        
        try
            img = imread(img_path);
            
            % Verificar que sigui una imatge vàlida
            if isempty(img) || ndims(img) < 3
                fprintf('  Imatge invàlida: %s\n', archivos(img_idx).name);
                total_errors = total_errors + 1;
                continue;
            end
            
            % Mostrar progrés cada 20 imatges
            if mod(img_idx, 20) == 0
                fprintf('  Processant imatge %d/%d\n', img_idx, length(archivos));
            end
            
            % Extreure descriptors jeràrquics
            [desc_forma, desc_color, desc_detall] = extractDescriptors4Models(img);
            
            % Verificar si TOTS els descriptors són exactament zero (error)
            if all(desc_forma == 0) && all(desc_color == 0) && all(desc_detall == 0)
                fprintf('  Descriptors buits (error): %s\n', archivos(img_idx).name);
                total_discarded = total_discarded + 1;
                continue;
            end
            
            % Verificar dimensions correctes
            if length(desc_forma) ~= 16
                fprintf('  Advertència: desc_forma mida incorrecta en %s (%d)\n', ...
                    archivos(img_idx).name, length(desc_forma));
                total_errors = total_errors + 1;
                continue; 
            end
            
            if length(desc_color) ~= 15
                fprintf('  Advertència: desc_color mida incorrecta en %s (%d)\n', ...
                    archivos(img_idx).name, length(desc_color));
                total_errors = total_errors + 1;
                continue;
            end
            
            if length(desc_detall) ~= 30
                fprintf('  Advertència: desc_detall mida incorrecta en %s (%d)\n', ...
                    archivos(img_idx).name, length(desc_detall));
                total_errors = total_errors + 1;
                continue;
            end
            
            % Determinar etiquetes segons categoria
            label_forma = '';
            label_color = '';
            skip_color_model = false;
            
            % Etiqueta per Model 1 (Forma)
            if strcmp(categoria, 'stop')
                label_forma = 'octagonal';
                skip_color_model = true;  % Excloure del model de color
                
            elseif strcmp(categoria, 'vianant')
                label_forma = 'triangular';
                skip_color_model = true;  % Excloure del model de color
                
            else
                label_forma = 'circular';
            end
            
            % Etiqueta per Model 2 (Color) - Només senyals circulars
            if ~skip_color_model
                if ismember(categoria, {'d_prohibida'})
                    label_color = 'red';
                elseif ismember(categoria, {'d_obligatoria', 'no_aparcar', 'zona_bici', 'zona_cotxe'})
                    label_color = 'blue';
                else
                    label_color = 'white';
                end
            end
            
            % Afegir a Model 1 (Formes) - Totes les senyals
            if ~isempty(desc_forma) && ~all(desc_forma == 0)
                descriptors_forma = [descriptors_forma; desc_forma];
                labels_forma = [labels_forma; label_forma];
            end
            
            % Afegir a Model 2 (Colors) - Excloure stop i vianant
            if ~isempty(desc_color) && ~skip_color_model && ~all(desc_color == 0)
                descriptors_color = [descriptors_color; desc_color];
                labels_color = [labels_color; label_color];
            end
            
            % Afegir a Models 3-4 segons tipus (només circulars)
            if ~isempty(desc_detall) && ~skip_color_model && ~all(desc_detall == 0)
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
            
            processed_in_cat = processed_in_cat + 1;
            total_processed = total_processed + 1;
            
        catch ME
            fprintf('  Error processant %s: %s\n', archivos(img_idx).name, ME.message);
            total_errors = total_errors + 1;
        end
    end
    
    fprintf('  Processades correctament: %d/%d\n', processed_in_cat, length(archivos));
end

%% CREAR I GUARDAR LES TAULES
fprintf('\n=== CREANT TAULES PER CLASSIFICATION LEARNER ===\n');

% Model 1: Formes
if ~isempty(descriptors_forma)
    train_table_forma = array2table(descriptors_forma, 'VariableNames', descriptor_names_forma);
    train_table_forma.Class = categorical(labels_forma);
    save('data_model1_forma.mat', 'train_table_forma');
    fprintf('Model 1 (Formes): %d mostres, %d classes\n', ...
        size(descriptors_forma, 1), length(unique(labels_forma)));
else
    fprintf('ADVERTÈNCIA: Model 1 (Formes) no té dades\n');
end

% Model 2: Colors
if ~isempty(descriptors_color)
    train_table_color = array2table(descriptors_color, 'VariableNames', descriptor_names_color);
    train_table_color.Class = categorical(labels_color);
    save('data_model2_color.mat', 'train_table_color');
    fprintf('Model 2 (Colors): %d mostres, %d classes\n', ...
        size(descriptors_color, 1), length(unique(labels_color)));
else
    fprintf('ADVERTÈNCIA: Model 2 (Colors) no té dades\n');
end

% Model 3: Circulars blaves
if ~isempty(descriptors_circ_azul)
    train_table_circ_azul = array2table(descriptors_circ_azul, 'VariableNames', descriptor_names_detall);
    train_table_circ_azul.Class = categorical(labels_circ_azul);
    save('data_model3_circ_azul.mat', 'train_table_circ_azul');
    fprintf('Model 3 (Circulars blaves): %d mostres, %d classes\n', ...
        size(descriptors_circ_azul, 1), length(unique(labels_circ_azul)));
else
    fprintf('ADVERTÈNCIA: Model 3 (Circulars blaves) no té dades\n');
end

% Model 4: Circulars blanques
if ~isempty(descriptors_circ_blanco)
    train_table_circ_blanco = array2table(descriptors_circ_blanco, 'VariableNames', descriptor_names_detall);
    train_table_circ_blanco.Class = categorical(labels_circ_blanco);
    save('data_model4_circ_blanco.mat', 'train_table_circ_blanco');
    fprintf('Model 4 (Circulars blanques): %d mostres, %d classes\n', ...
        size(descriptors_circ_blanco, 1), length(unique(labels_circ_blanco)));
else
    fprintf('ADVERTÈNCIA: Model 4 (Circulars blanques) no té dades\n');
end

%% RESUM FINAL
fprintf('\n=== RESULTATS FINALS ===\n');
fprintf('Total d''imatges processades correctament: %d\n', total_processed);
fprintf('Total d''errors: %d\n', total_errors);
fprintf('Total descartades (descriptors buits): %d\n', total_discarded);
fprintf('Total real d''errors: %d\n', total_errors + total_discarded);

if total_processed == 0
    fprintf('\nERROR: No s''ha processat cap imatge correctament\n');
else
    fprintf('\n✓ Taules generades correctament per als 4 models\n');
end