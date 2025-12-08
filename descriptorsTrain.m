%% PRÀCTICA VC - RECONEIXEMENT DE SENYALS DE TRÀNSIT
% Extracció de descriptors de train

clear; clc; close all;

%% CONFIGURACIÓ
train_path = 'imatges_senyals/train';
categorias = {
    'd_obligatoria',        % circular, azul, flecha blanca
    'd_prohibida',          % circular, roja con franja blanca  
    'limit',                % circular, blanca y roja, flecha negra tachada
    'no_aparcar',           % circular, blanca y roja con numero negro
    'no_girar',             % octogonal, roja con letras blancas
    'no_soroll',            % triangular, amarilla con señor negro
    'stop',                 % redonda, azul y roja, cruz roja en fondo azul
    'vianant',              % redonda, azul con bici blanca
    'zona_bici',            % redonda, blanca y roja, flecha negra tachada de rojo
    'zona_cotxe'            % redonda, azul con coche blanco
};

%% VERIFICAR ESTRUCTURA DE CARPETES
fprintf('=== VERIFICANT ESTRUCTURA DE CARPETES ===\n');
fprintf('Ruta base: %s\n', train_path);

if ~exist(train_path, 'dir')
    fprintf('ERROR: No existeix el directori: %s\n', train_path);
    return;
end

%% GENERAR LA TAULA DE DESCRIPTORS PER TOTES LES IMATGES
fprintf('\n=== EXTRACCIÓ DE DESCRIPTORS ===\n');

% Inicialitzar
all_descriptors = [];
all_labels = {};

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

total_processed = 0;
total_errors = 0;

for cat_idx = 1:length(categorias)
    categoria = categorias{cat_idx};
    carpeta = fullfile(train_path, categoria);
    
    % Verificar que existeix la carpeta
    if ~exist(carpeta, 'dir')
        fprintf('ADVERTÈNCIA: No existeix la carpeta: %s\n', carpeta);
        continue;
    end
    
    fprintf('Processant categoria: %s\n', categoria);
    
    % Llistar imatges
    archivos = dir(fullfile(carpeta, '*.png'));
    if isempty(archivos)
        archivos = dir(fullfile(carpeta, '*.jpg'));
    end
    if isempty(archivos)
        archivos = dir(fullfile(carpeta, '*.jpeg'));
    end
    
    if isempty(archivos)
        fprintf('  No s''han trovat imatges en %s\n', carpeta);
        continue;
    end
    
    fprintf('  Trovades %d imatges\n', length(archivos));
    
    processed_in_cat = 0;
    
    for img_idx = 1:length(archivos)
        % Carregar imatge
        img_path = fullfile(carpeta, archivos(img_idx).name);
        
        try
            img = imread(img_path);
            
            % Verificar que sea una imagen válida
            if isempty(img) || ndims(img) < 3
                fprintf('  Imatge invàlida: %s\n', archivos(img_idx).name);
                continue;
            end
            
            % Mostrar progrés
            if mod(img_idx, 20) == 0
                fprintf('  Processant imatge %d/%d\n', img_idx, length(archivos));
            end
            
            % Extreure descriptors
            desc = extractDescriptors(img);
            
            % Verificar que los descriptores tengan la dimensión correcta
            if length(desc) == 25
                % Afegir a les llistes
                all_descriptors = [all_descriptors; desc];
                all_labels = [all_labels; categoria];
                processed_in_cat = processed_in_cat + 1;
                total_processed = total_processed + 1;
            else
                fprintf('  ERROR: Descriptors incorrectes en %s (tamaño: %d)\n', ...
                    archivos(img_idx).name, length(desc));
                total_errors = total_errors + 1;
            end
            
        catch ME
            fprintf('  Error processant %s: %s\n', archivos(img_idx).name, ME.message);
            total_errors = total_errors + 1;
        end
    end
    
    fprintf('  Processades correctament: %d/%d\n', processed_in_cat, length(archivos));
end

%% CREAR I GUARDAR LA TAULA PER CLASSIFICATION LEARNER
fprintf('\n=== RESULTATS FINALS ===\n');
fprintf('Total d''imatges processades correctament: %d\n', total_processed);
fprintf('Total d''errors: %d\n', total_errors);

if total_processed == 0
    fprintf('ERROR: No s''ha processat cap imatge\n');
    return;
end

fprintf('Total de descriptors per imatge: %d\n', size(all_descriptors, 2));

% Crear taula
train_table = array2table(all_descriptors, 'VariableNames', descriptor_names);
train_table.Class = categorical(all_labels);

% Guardar
save('train_data_table.mat', 'train_table');
fprintf('\nTaula guardada com: train_data_table.mat\n');