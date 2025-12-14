%% ajustar_umbrales.m
% Script para encontrar los umbrales óptimos visualmente

clear; clc; close all;

%% CARGAR IMAGEN
train_path = 'imatges_senyals/train';
categoria = 'limit';
nombre_archivo = 'road869.png';

img_path = fullfile(train_path, categoria, nombre_archivo);
if ~exist(img_path, 'file')
    fprintf('ERROR: No es troba la imatge\n');
    return;
end

img = imread(img_path);
[rows, cols, ~] = size(img);

%% CONVERSIÓN DE ESPACIOS DE COLOR
img_hsv = rgb2hsv(img);
H = img_hsv(:,:,1);
S = img_hsv(:,:,2);
V = img_hsv(:,:,3);

% Normalizar
normalized_hsv = cat(3, H, S, ones(rows, cols));
rgb_norm = hsv2rgb(normalized_hsv);
R = rgb_norm(:,:,1);
G = rgb_norm(:,:,2);
B = rgb_norm(:,:,3);

%% PRUEBA DE DIFERENTES NIVELES DE UMBRAL

figure('Name', 'Comparación de Umbrales', 'Position', [50, 50, 1600, 900]);

% ========================================
% NIVEL 1: TU CÓDIGO ORIGINAL (Permisivo)
% ========================================
subplot(3,4,1), imshow(img), title('Original');

% Vermells - Original
red1_orig = (R > 0.5) & (G < 0.5) & (B < 0.5);
red2_orig = (H > 0.95 | H < 0.05) & (S > 0.5) & (V > 0.4);

% Blaus - Original
blue1_orig = (B > 0.5) & (R < 0.3) & (G < 0.5);
blue2_orig = (H > 0.55 & H < 0.7) & (S > 0.4) & (V > 0.3);

% Grocs - Original
yellow1_orig = (R > 0.5) & (G > 0.5) & (B < 0.3);
yellow2_orig = (H > 0.1 & H < 0.2) & (S > 0.4) & (V > 0.4);

orange1_orig = (R > 0.8) & (G > 0.4) & (B < 0.3);
orange2_orig = (H > 0.05 & H < 0.1) & (S > 0.4) & (V > 0.4);

colorMask_orig = red1_orig | red2_orig | blue1_orig | blue2_orig | ...
                 yellow1_orig | yellow2_orig | orange1_orig | orange2_orig;

subplot(3,4,2), imshow(colorMask_orig), title('Nivel 1: Original (Permisivo)');

% ========================================
% NIVEL 2: EQUILIBRADO (Recomendado)
% ========================================

% Vermells - Equilibrado
red1_bal = (R > 0.55) & (G < 0.45) & (B < 0.45);
red2_bal = (H > 0.95 | H < 0.05) & (S > 0.45) & (V > 0.35);

% Blaus - Equilibrado
blue1_bal = (B > 0.55) & (R < 0.35) & (G < 0.48);
blue2_bal = (H > 0.56 & H < 0.69) & (S > 0.4) & (V > 0.3);

% Grocs - Equilibrado
yellow1_bal = (R > 0.55) & (G > 0.5) & (B < 0.3);
yellow2_bal = (H > 0.11 & H < 0.19) & (S > 0.45) & (V > 0.4);

orange1_bal = (R > 0.75) & (G > 0.35 & G < 0.65) & (B < 0.3);
orange2_bal = (H > 0.05 & H < 0.1) & (S > 0.45) & (V > 0.4);

colorMask_bal = red1_bal | red2_bal | blue1_bal | blue2_bal | ...
                yellow1_bal | yellow2_bal | orange1_bal | orange2_bal;

% AÑADIR filtro de saturación suave
saturation_filter = S > 0.25;  % Más permisivo que 0.3
colorMask_bal = colorMask_bal & saturation_filter;

subplot(3,4,3), imshow(colorMask_bal), title('Nivel 2: Equilibrado');

% ========================================
% NIVEL 3: ANALIZAR POR SEPARADO
% ========================================
subplot(3,4,4), imshow(red1_bal | red2_bal), title('Solo Rojos');
subplot(3,4,5), imshow(blue1_bal | blue2_bal), title('Solo Azules');
subplot(3,4,6), imshow(yellow1_bal | yellow2_bal | orange1_bal | orange2_bal), ...
    title('Solo Amarillos/Naranjas');

%% PROCESAMIENTO COMPLETO CON NIVEL EQUILIBRADO

% Morfología
ee_small = strel('disk', 1);
ee_medium = strel('disk', 2);

morphMask = imclose(colorMask_bal, ee_small);
morphMask = imopen(morphMask, ee_small);
morphMask = imclose(morphMask, ee_medium);

subplot(3,4,7), imshow(morphMask), title('Después de Morfología');

% Aplicar máscara y detectar edges
img_masked = img;
for canal = 1:3
    ch = img_masked(:,:,canal);
    ch(~morphMask) = 0;
    img_masked(:,:,canal) = ch;
end

img_gray_masked = rgb2gray(img_masked);
edges = edge(img_gray_masked, 'Canny', [0.08 0.18]);  % Umbral ligeramente más bajo

subplot(3,4,8), imshow(edges), title('Edges');

% Combinar
combi = morphMask | edges;
combi = imclose(combi, strel('disk', 3));
combi = imfill(combi, 'holes');
combi = bwareaopen(combi, 150);  % Más permisivo (antes 200)

subplot(3,4,9), imshow(combi), title('Combinado');

%% SELECCIÓN DE REGIÓN (MÁS PERMISIVA)

stats = regionprops(combi, 'Area', 'BoundingBox', 'Solidity', ...
                    'PixelIdxList', 'Centroid', 'Eccentricity');

if ~isempty(stats)
    % Criterios MÁS PERMISIVOS
    valid = false(length(stats), 1);
    
    for i = 1:length(stats)
        area = stats(i).Area;
        solidity = stats(i).Solidity;
        bbox = stats(i).BoundingBox;
        aspect_ratio = bbox(3) / bbox(4);
        
        % Criterios ajustados:
        area_ok = area > 200 && area < (rows * cols * 0.85);  % Más permisivo
        solidity_ok = solidity > 0.6;  % Más permisivo (antes 0.65)
        aspect_ok = aspect_ratio > 0.3 && aspect_ratio < 3;  % Más rango
        
        valid(i) = area_ok && solidity_ok && aspect_ok;
        
        % Debug: mostrar por qué se rechaza
        if ~valid(i) && area > 100
            fprintf('Región %d rechazada: Area=%.0f Solidity=%.2f AspectRatio=%.2f\n', ...
                    i, area, solidity, aspect_ratio);
        end
    end
    
    stats = stats(valid);
    
    if ~isempty(stats)
        % Seleccionar la más grande
        [~, idx] = max([stats.Area]);
        best = stats(idx);
        
        % Crear máscara
        mascara_final = false(rows, cols);
        mascara_final(best.PixelIdxList) = true;
        mascara_final = imclose(mascara_final, strel('disk', 2));
        mascara_final = imfill(mascara_final, 'holes');
        
        subplot(3,4,10), imshow(mascara_final), title('Región Principal');
        
        % Extraer ROI
        bbox = best.BoundingBox;
        x = max(1, round(bbox(1)));
        y = max(1, round(bbox(2)));
        w = min(round(bbox(3)), cols - x + 1);
        h = min(round(bbox(4)), rows - y + 1);
        
        if w > 0 && h > 0
            img_roi = img(y:y+h-1, x:x+w-1, :);
            
            % Aplicar máscara a imagen original
            img_final = img;
            for canal = 1:3
                ch = img_final(:,:,canal);
                ch(~mascara_final) = 0;
                img_final(:,:,canal) = ch;
            end
            
            subplot(3,4,11), imshow(img_roi), title('ROI Extraída');
            subplot(3,4,12), imshow(img_final), title('Resultado Final');
            
            fprintf('\n=== REGIÓN SELECCIONADA ===\n');
            fprintf('Área: %d píxels\n', best.Area);
            fprintf('Solidesa: %.2f\n', best.Solidity);
            fprintf('BBox: [%.0f, %.0f, %.0f, %.0f]\n', bbox);
        end
    else
        fprintf('No se encontraron regiones válidas\n');
        subplot(3,4,10), imshow(combi), title('Sin filtrar');
    end
else
    fprintf('No se detectaron regiones\n');
end

%% COMPARACIÓN LADO A LADO
figure('Name', 'Comparación Final', 'Position', [100, 100, 1200, 400]);
subplot(1,3,1), imshow(img), title('Original');
subplot(1,3,2), imshow(colorMask_orig), title('Método Original');
subplot(1,3,3), imshow(colorMask_bal), title('Método Equilibrado');