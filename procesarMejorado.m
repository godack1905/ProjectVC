%% PRÀCTICA VC - RECONEIXEMENT DE SENYALS DE TRÀNSIT
% Processament d'imatges (s'ha inclos a extractDescriptors)
%clear; clc; close all;

train_path = 'imatges_senyals/train';

% Definim la categoria i el nom de la imatge a processar

categoria = 'no_girar';
nombre_archivo = '011_1_0002.png';

%% 0. Carreguem la imatge
img_path = fullfile(train_path, categoria, nombre_archivo);
if ~exist(img_path, 'file')
     fprintf('ERROR: No es troba la imatge: %s\n', img_path);
    return;
end

img = imread(img_path);
[rows, cols, ~] = size(img);

figure('Position', [50, 50, 1400, 800]);
subplot(2,4,1), imshow(img), title('1. Original');

%% 1. PROCESAMENT DE COLORS
img_gray = rgb2gray(img);

img_hsv = rgb2hsv(img);
H = img_hsv(:,:,1);
S = img_hsv(:,:,2);
V = img_hsv(:,:,3);

% Normalitzar
VAL = double(ones(rows, cols));
normalized_hsv = cat(3, H, S, VAL);
rgb_norm = hsv2rgb(normalized_hsv);
R = rgb_norm(:,:,1);
G = rgb_norm(:,:,2);
B = rgb_norm(:,:,3);

subplot(2,4,2), imshow(rgb_norm), title('2. Normalitzat');

%% 2. Detecció per color

% Detecció de vermells
red_rgb = (R > 0.55) & (G < 0.45) & (B < 0.45);
red_hsv = (H > 0.95 | H < 0.05) & (S > 0.45) & (V > 0.35);
red_mask = red_rgb | red_hsv;

% Detecció de blaus
blue_rgb = (B > 0.55) & (R < 0.35) & (G < 0.48);
blue_hsv = (H > 0.56 & H < 0.69) & (S > 0.4) & (V > 0.3);
blue_mask = blue_rgb | blue_hsv;

% Detecció de grocs i taronjas
yellow_rgb = (R > 0.55) & (G > 0.5) & (B < 0.3);
yellow_hsv = (H > 0.11 & H < 0.19) & (S > 0.45) & (V > 0.4);
orange_rgb = (R > 0.75) & (G > 0.35 & G < 0.65) & (B < 0.3);
orange_hsv = (H > 0.05 & H < 0.1) & (S > 0.45) & (V > 0.4);
yellow_mask = yellow_rgb | yellow_hsv | orange_rgb | orange_hsv;

% Combinem els colors
colorMask = red_mask | blue_mask | yellow_mask;

% Filtro de saturación (eliminar colores apagados del fondo)
saturation_filter = S > 0.35;
colorMask = colorMask & saturation_filter;

subplot(2,4,3), imshow(colorMask), title('3. Colors detectats');

%% 3. Operacions morfològiques
ee1 = strel('disk', 1);
ee2 = strel('disk', 2);

morphMask = imfill(colorMask, 'holes');

morphMask = imclose(morphMask, ee2);
morphMask = imopen(morphMask, ee1);
morphMask = bwareaopen(morphMask, 150);

subplot(2,4,4), imshow(morphMask), title('4. Morfologia');

%% 4. Detecció d'edges
img_masked = img;
for c = 1:3
    ch = img_masked(:,:,c);
    ch(~morphMask) = 0;
    img_masked(:,:,c) = ch;
end

edges = edge(rgb2gray(img_masked), 'Canny', [0.08 0.18]);
subplot(2,4,5), imshow(edges), title('5. Edges Canny');

% Combinar color i edges
combi = morphMask | edges;
combi = imclose(combi, strel('disk', 1));
combi = imfill(combi, 'holes');
combi = bwareaopen(combi, 150);

subplot(2,4,6), imshow(combi), title('6. Combinat');

%% 5. Extracció de contorns
stats = regionprops(combi, 'Area', 'BoundingBox', 'Solidity', ...
                    'PixelIdxList', 'Eccentricity');

if isempty(stats)
    fprintf('No s''han detectat regions\n');
    img_final = img;
else
    % Filtrar regions
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
        fprintf('No hi ha regions valides\n');
        img_final = img;
        mascara_final = combi;
    else
        % Regio mes gran
        [~, idx] = max([stats.Area]);
        best = stats(idx);
        
        mascara_final = false(rows, cols);
        mascara_final(best.PixelIdxList) = true;
        mascara_final = imclose(mascara_final, strel('disk', 2));
        mascara_final = imfill(mascara_final, 'holes');
        
        subplot(2,4,7), imshow(mascara_final), title('7. Regió principal');
        
        % Aplicar máscara
        img_final = img;
        for c = 1:3
            ch = img_final(:,:,c);
            ch(~mascara_final) = 0;
            img_final(:,:,c) = ch;
        end
        
        % Extraer ROI
        bbox = best.BoundingBox;
        x = max(1, round(bbox(1)));
        y = max(1, round(bbox(2)));
        w = min(round(bbox(3)), cols - x + 1);
        h = min(round(bbox(4)), rows - y + 1);
        
        if w > 0 && h > 0
            img_roi = img(y:y+h-1, x:x+w-1, :);
            
            % Visualizar
            img_bbox = insertShape(img, 'Rectangle', bbox, ...
                                   'Color', 'green', 'LineWidth', 3);
            subplot(2,4,8), imshow(img_bbox), title('8. Detecció');
            
            fprintf('\n✅ REGIÓ DETECTADA\n');
            fprintf('Àrea: %d px (%.1f%% imatge)\n', best.Area, ...
                    best.Area/(rows*cols)*100);
            fprintf('Solidesa: %.2f\n', best.Solidity);
        end
    end
end

% Mostrar resultado final
figure;
subplot(1,2,1), imshow(img), title('Original');
subplot(1,2,2), imshow(img_final), title('Processat');