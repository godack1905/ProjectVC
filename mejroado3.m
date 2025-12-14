%% procesament_v2_multiples_senyals.m
clear; clc; close all;

train_path = 'imatges_senyals/train';
categoria = 'limit';
nombre_archivo = 'road869.png';

%% CARGAR IMAGEN
img_path = fullfile(train_path, categoria, nombre_archivo);
if ~exist(img_path, 'file')
    fprintf('ERROR: No es troba la imatge\n');
    return;
end

img = imread(img_path);
[rows, cols, ~] = size(img);

figure('Position', [50, 50, 1400, 800]);
subplot(2,4,1), imshow(img), title('1. Original');

%% CONVERSIÓN DE COLOR
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

subplot(2,4,2), imshow(rgb_norm), title('2. Normalitzat');

%% DETECCIÓN DE COLOR - UMBRALES EQUILIBRADOS

% ROJOS (señales de stop, prohibición, límites)
red_rgb = (R > 0.55) & (G < 0.45) & (B < 0.45);
red_hsv = (H > 0.95 | H < 0.05) & (S > 0.45) & (V > 0.35);
red_mask = red_rgb | red_hsv;

% AZULES (señales obligatorias, información)
blue_rgb = (B > 0.55) & (R < 0.35) & (G < 0.48);
blue_hsv = (H > 0.56 & H < 0.69) & (S > 0.4) & (V > 0.3);
blue_mask = blue_rgb | blue_hsv;

% AMARILLOS Y NARANJAS (advertencia)
yellow_rgb = (R > 0.55) & (G > 0.5) & (B < 0.3);
yellow_hsv = (H > 0.11 & H < 0.19) & (S > 0.45) & (V > 0.4);
orange_rgb = (R > 0.75) & (G > 0.35 & G < 0.65) & (B < 0.3);
orange_hsv = (H > 0.05 & H < 0.1) & (S > 0.45) & (V > 0.4);
yellow_mask = yellow_rgb | yellow_hsv | orange_rgb | orange_hsv;

% Combinar todos los colores
colorMask = red_mask | blue_mask | yellow_mask;

% Filtro de saturación
saturation_filter = S > 0.25;
colorMask = colorMask & saturation_filter;

subplot(2,4,3), imshow(colorMask), title('3. Colors detectats');

%% MORFOLOGÍA
ee1 = strel('disk', 1);
ee2 = strel('disk', 2);

morphMask = imclose(colorMask, ee1);
morphMask = imopen(morphMask, ee1);
morphMask = imclose(morphMask, ee2);

subplot(2,4,4), imshow(morphMask), title('4. Morfologia');

%% DETECCIÓN DE BORDES
img_masked = img;
for c = 1:3
    ch = img_masked(:,:,c);
    ch(~morphMask) = 0;
    img_masked(:,:,c) = ch;
end

edges = edge(rgb2gray(img_masked), 'Canny', [0.08 0.18]);
subplot(2,4,5), imshow(edges), title('5. Edges');

%% COMBINACIÓN
combi = morphMask | edges;
combi = imclose(combi, strel('disk', 3));
combi = imfill(combi, 'holes');
combi = bwareaopen(combi, 150);

subplot(2,4,6), imshow(combi), title('6. Combinat');

%% SELECCIÓN DE TODAS LAS REGIONES VÁLIDAS (CAMBIO PRINCIPAL)
stats = regionprops(combi, 'Area', 'BoundingBox', 'Solidity', ...
                    'PixelIdxList', 'Eccentricity');

if isempty(stats)
    fprintf('⚠️ No s''han detectat regions\n');
    img_final = img;
    detected_signals = [];
else
    % Filtrar regiones válidas
    valid = false(length(stats), 1);
    for i = 1:length(stats)
        area_ok = stats(i).Area > 200 && stats(i).Area < rows*cols*0.85;
        solidity_ok = stats(i).Solidity > 0.6;
        bbox = stats(i).BoundingBox;
        aspect_ok = (bbox(3)/bbox(4)) > 0.3 && (bbox(3)/bbox(4)) < 3;
        
        valid(i) = area_ok && solidity_ok && aspect_ok;
    end
    
    stats = stats(valid);
    
    if isempty(stats)
        fprintf('⚠️ No hi ha regions vàlides\n');
        img_final = img;
        detected_signals = [];
    else
        % PROCESAR TODAS LAS REGIONES VÁLIDAS (no solo la más grande)
        num_signals = length(stats);
        fprintf('\n✅ DETECTADES %d SENYALS\n', num_signals);
        fprintf('═══════════════════════════════════\n');
        
        % Crear máscara combinada
        mascara_final = false(rows, cols);
        img_bbox = img;
        detected_signals = cell(num_signals, 1);
        
        % Ordenar por área (más grande primero) para visualización
        [~, sort_idx] = sort([stats.Area], 'descend');
        stats = stats(sort_idx);
        
        for i = 1:num_signals
            region = stats(i);
            
            % Añadir región a la máscara
            mascara_region = false(rows, cols);
            mascara_region(region.PixelIdxList) = true;
            mascara_region = imclose(mascara_region, strel('disk', 2));
            mascara_region = imfill(mascara_region, 'holes');
            mascara_final = mascara_final | mascara_region;
            
            % Extraer bounding box
            bbox = region.BoundingBox;
            x = max(1, round(bbox(1)));
            y = max(1, round(bbox(2)));
            w = min(round(bbox(3)), cols - x + 1);
            h = min(round(bbox(4)), rows - y + 1);
            
            if w > 0 && h > 0
                % Extraer ROI de cada señal
                img_roi = img(y:y+h-1, x:x+w-1, :);
                detected_signals{i} = img_roi;
                
                % Dibujar bounding box
                colors = {'green', 'blue', 'red', 'yellow', 'cyan', 'magenta'};
                color = colors{mod(i-1, 6) + 1};
                img_bbox = insertShape(img_bbox, 'Rectangle', bbox, ...
                                       'Color', color, 'LineWidth', 3);
                img_bbox = insertText(img_bbox, [bbox(1), bbox(2)-20], ...
                                      sprintf('#%d', i), 'FontSize', 18, ...
                                      'BoxColor', color, 'BoxOpacity', 0.8);
                
                % Mostrar info de cada señal
                fprintf('Senyal #%d:\n', i);
                fprintf('  Àrea: %d px (%.1f%% imatge)\n', region.Area, ...
                        region.Area/(rows*cols)*100);
                fprintf('  Solidesa: %.2f\n', region.Solidity);
                fprintf('  Posició: [%d, %d], Mida: %dx%d\n', x, y, w, h);
                fprintf('  ───────────────────\n');
            end
        end
        
        subplot(2,4,7), imshow(mascara_final), title('7. Totes les regions');
        subplot(2,4,8), imshow(img_bbox), title(sprintf('8. %d senyals detectades', num_signals));
        
        % Aplicar máscara final
        img_final = img;
        for c = 1:3
            ch = img_final(:,:,c);
            ch(~mascara_final) = 0;
            img_final(:,:,c) = ch;
        end
        
        % MOSTRAR CADA SEÑAL DETECTADA EN VENTANA SEPARADA
        if num_signals > 0
            figure('Name', 'Senyals detectades individualment');
            n_cols = min(num_signals, 4);
            n_rows = ceil(num_signals / n_cols);
            
            for i = 1:num_signals
                subplot(n_rows, n_cols, i);
                imshow(detected_signals{i});
                title(sprintf('Senyal #%d', i));
            end
        end
    end
end

% Resultado final
figure('Name', 'Resultat final');
subplot(1,2,1), imshow(img), title('Original');
subplot(1,2,2), imshow(img_final), title(sprintf('Processat (%d senyals)', length(stats)));