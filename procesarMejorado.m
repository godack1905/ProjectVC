%% SISTEMA ROBUSTO DE DETECCIÓN DE SEÑALES - FUSIÓN INTELIGENTE
% Combina color y forma de manera adaptativa para máxima robustez
clear; clc; close all;

train_path = 'imatges_senyals/train';
categoria = 'vianant';
nombre_archivo = '035_1_0002.png';

%% Cargar imagen
img_path = fullfile(train_path, categoria, nombre_archivo);
if ~exist(img_path, 'file')
    fprintf('ERROR: No se encuentra la imagen: %s\n', img_path);
    return;
end

img = imread(img_path);
[rows, cols, ~] = size(img);

figure('Position', [50, 50, 1600, 900]);
subplot(3,4,1), imshow(img), title('1. Original');

%% 1. PROCESAMIENTO DE COLORES
img_gray = rgb2gray(img);

img_hsv = rgb2hsv(img);
H = img_hsv(:,:,1);
S = img_hsv(:,:,2);
V = img_hsv(:,:,3);

% Normalizar
VAL = double(ones(rows, cols));
normalized_hsv = cat(3, H, S, VAL);
rgb_norm = hsv2rgb(normalized_hsv);
R = rgb_norm(:,:,1);
G = rgb_norm(:,:,2);
B = rgb_norm(:,:,3);

subplot(3,4,2), imshow(rgb_norm), title('2. Normalizado');

%% 2. DETECCIÓN POR COLOR - MEJORADA
% Detección de rojos (más permisiva)
red_rgb = (R > 0.5) & (G < 0.5) & (B < 0.5);
red_hsv = (H > 0.94 | H < 0.06) & (S > 0.4) & (V > 0.3);
red_mask = red_rgb | red_hsv;

% Detección de azules (más permisiva)
blue_rgb = (B > 0.5) & (R < 0.4) & (G < 0.5);
blue_hsv = (H > 0.54 & H < 0.70) & (S > 0.35) & (V > 0.25);
blue_mask = blue_rgb | blue_hsv;

% Detección de amarillos y naranjas
yellow_rgb = (R > 0.5) & (G > 0.45) & (B < 0.35);
yellow_hsv = (H > 0.10 & H < 0.20) & (S > 0.4) & (V > 0.35);
orange_rgb = (R > 0.7) & (G > 0.3 & G < 0.7) & (B < 0.35);
orange_hsv = (H > 0.04 & H < 0.11) & (S > 0.4) & (V > 0.35);
yellow_mask = yellow_rgb | yellow_hsv | orange_rgb | orange_hsv;

% Combinar colores
colorMask = red_mask | blue_mask | yellow_mask;

% Filtro de saturación para eliminar fondos apagados
saturation_filter = S > 0.3;
colorMask = colorMask & saturation_filter;

subplot(3,4,3), imshow(colorMask), title('3. Colores detectados');

%% 3. DETECCIÓN DE FORMAS
img_enhanced = imadjust(img_gray);
edges_all = edge(img_enhanced, 'Canny', [0.08 0.22]);

subplot(3,4,4), imshow(edges_all), title('4. Edges Canny');

% Inicializar máscaras
circle_mask = false(rows, cols);
triangle_mask = false(rows, cols);
octagon_mask = false(rows, cols);

%% A) DETECCIÓN DE CÍRCULOS CON FILTRADO INTELIGENTE
radio_min = 15;
radio_max = round(min(rows, cols) / 2);
[centers, radii, metric] = imfindcircles(img_enhanced, [radio_min radio_max], ...
    'ObjectPolarity', 'dark', ...
    'Sensitivity', 0.93, ...
    'EdgeThreshold', 0.08);

% GUARDAR TODOS LOS CÍRCULOS DETECTADOS ORIGINALMENTE
all_centers = centers;
all_radii = radii;
all_metric = metric;

num_circles = 0;
circle_centers = [];
circle_radii = [];

if ~isempty(centers)
    fprintf('\n=== ANÁLISIS DE CÍRCULOS DETECTADOS ===\n');
    fprintf('Círculos encontrados: %d\n', length(radii));
    
    % FILTRAR: Quedarse solo con el círculo más prometedor si hay varios
    % Criterios: mayor métrica, mayor radio, más centrado
    
    if length(radii) > 1
        fprintf('Aplicando filtrado de círculos...\n');
        
        % Calcular scores para cada círculo
        center_img = [cols/2, rows/2];
        circle_scores = zeros(length(radii), 1);
        
        for i = 1:length(radii)
            % Score basado en: métrica de detección + tamaño + centralidad
            dist_to_center = norm([centers(i,1), centers(i,2)] - center_img);
            max_dist = norm(center_img);
            centrality = 1 - (dist_to_center / max_dist);
            
            % Verificar solapamiento con máscara de color
            [xx, yy] = meshgrid(1:cols, 1:rows);
            circle_temp = ((xx - centers(i,1)).^2 + (yy - centers(i,2)).^2) <= radii(i)^2;
            color_overlap = sum(colorMask(:) & circle_temp(:)) / sum(circle_temp(:));
            
            % Score final: métrica + tamaño normalizado + centralidad + color
            circle_scores(i) = metric(i) * 0.3 + ...
                              (radii(i)/radio_max) * 0.2 + ...
                              centrality * 0.2 + ...
                              color_overlap * 0.3;
            
            fprintf('  Círculo %d: Radio=%.1f, Centro=(%.1f,%.1f), Métrica=%.2f, ColorOverlap=%.2f, Score=%.3f\n', ...
                i, radii(i), centers(i,1), centers(i,2), metric(i), color_overlap, circle_scores(i));
        end
        
        % Seleccionar solo el mejor círculo
        [~, best_idx] = max(circle_scores);
        fprintf('✓ Seleccionado círculo #%d (mejor score)\n', best_idx);
        
        centers = centers(best_idx, :);
        radii = radii(best_idx);
        metric = metric(best_idx);
    end
    
    num_circles = length(radii);
    circle_centers = centers;
    circle_radii = radii;
    
    % Crear máscara solo del círculo seleccionado - EXPANSIÓN MÍNIMA
    for i = 1:num_circles
        [xx, yy] = meshgrid(1:cols, 1:rows);
        circle_temp = ((xx - centers(i,1)).^2 + (yy - centers(i,2)).^2) <= (radii(i)*1.02)^2; % 2% expansión (era 1.2 = 20%)
        circle_mask = circle_mask | circle_temp;
    end
    
    % MOSTRAR TODOS LOS CÍRCULOS DETECTADOS (NO SOLO EL ELEGIDO)
    subplot(3,4,5);
    imshow(img);
    if ~isempty(all_centers)
        viscircles(all_centers, all_radii, 'Color', 'r', 'LineWidth', 1.5);
        % Marcar el círculo seleccionado en verde
        viscircles(centers, radii, 'Color', 'g', 'LineWidth', 2);
    end
    title(sprintf('5. Círculos: %d total, 1 seleccionado', length(all_radii)));
else
    subplot(3,4,5), imshow(img), title('5. Círculos: 0');
end

%% B) DETECCIÓN DE TRIÁNGULOS - ESTRATEGIA HÍBRIDA
fprintf('\n=== ANÁLISIS DE TRIÁNGULOS (HÍBRIDO) ===\n');

% ESTRATEGIA: Partir de regiones de COLOR (rellenas), luego analizar sus contornos

% 1. Preparar máscara de color limpia y rellenada
colorMask_clean = imclose(colorMask, strel('disk', 4));
colorMask_clean = imfill(colorMask_clean, 'holes');
colorMask_clean = bwareaopen(colorMask_clean, 200);

% 2. Preparar edges para validación
colorMask_dilated = imdilate(colorMask, strel('disk', 8));
edges_masked = edges_all & colorMask_dilated;
subplot(3,4,6), imshow(colorMask_clean), title('6. Regiones de color');

% 3. Extraer regiones de color como punto de partida
CC = bwconncomp(colorMask_clean);
num_regions = CC.NumObjects;

fprintf('Regiones de color encontradas: %d\n', num_regions);

num_triangles = 0;
num_octagons = 0;

% 4. Analizar cada región de color
for k = 1:num_regions
    % Crear máscara de esta región
    mask_region = false(rows, cols);
    mask_region(CC.PixelIdxList{k}) = true;
    
    % Calcular propiedades básicas
    stats = regionprops(mask_region, 'Area', 'Perimeter', 'BoundingBox', ...
        'Solidity', 'Extent', 'ConvexArea', 'Eccentricity', 'MajorAxisLength', ...
        'MinorAxisLength', 'Centroid');
    
    if isempty(stats)
        continue;
    end
    
    area = stats(1).Area;
    perimeter = stats(1).Perimeter;
    bbox = stats(1).BoundingBox;
    
    % Filtros básicos
    if area < 300 || area > rows*cols*0.75
        continue;
    end
    
    if bbox(3) < 15 || bbox(4) < 15
        continue;
    end
    
    % Métricas geométricas
    solidity = stats(1).Solidity;
    extent = stats(1).Extent;
    convex_area = stats(1).ConvexArea;
    eccentricity = stats(1).Eccentricity;
    
    if perimeter > 0
        form_factor = (4 * pi * area) / (perimeter^2);
    else
        continue;
    end
    
    convexity = area / convex_area;
    aspect_ratio = bbox(3) / bbox(4);
    
    % ═══════════════════════════════════════════════════════════
    % MÉTODO 1: ANÁLISIS DEL CONTORNO EXTERIOR
    % ═══════════════════════════════════════════════════════════
    
    is_triangle_contour = false;
    best_vertices = 0;
    best_tolerance = 0;
    
    % Extraer contorno exterior
    boundaries = bwboundaries(mask_region, 'noholes');
    
    if ~isempty(boundaries)
        boundary = boundaries{1};
        
        % Probar múltiples tolerancias para simplificación
        for tolerance = [0.008, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
            simplified = reducepoly(boundary, tolerance);
            num_vertices = size(simplified, 1);
            
            % TRIÁNGULOS: 3-6 vértices
            if num_vertices >= 3 && num_vertices <= 6
                % Criterios geométricos para triángulos
                ff_ok = (form_factor >= 0.38 && form_factor <= 0.75);
                sol_ok = (solidity >= 0.70 && solidity <= 0.96);
                conv_ok = (convexity >= 0.75);
                asp_ok = (aspect_ratio >= 0.60 && aspect_ratio <= 1.50);
                ecc_ok = (eccentricity <= 0.70);
                ext_ok = (extent >= 0.35);
                
                % Contar criterios cumplidos
                score = ff_ok + sol_ok + conv_ok + asp_ok + ecc_ok + ext_ok;
                
                % Necesitamos al menos 4 de 6 criterios
                if score >= 4
                    is_triangle_contour = true;
                    best_vertices = num_vertices;
                    best_tolerance = tolerance;
                    break;
                end
            end
        end
    end
    
    % ═══════════════════════════════════════════════════════════
    % MÉTODO 2: ANÁLISIS DE ESQUINAS
    % ═══════════════════════════════════════════════════════════
    
    is_triangle_corners = false;
    num_corners = 0;
    
    if ~is_triangle_contour && area > 400
        % Detectar esquinas en la región
        mask_uint8 = uint8(mask_region) * 255;
        corners = detectHarrisFeatures(mask_uint8, 'MinQuality', 0.02);
        num_corners = corners.Count;
        
        % Triángulos: 3-7 esquinas (con margen por ruido)
        if num_corners >= 3 && num_corners <= 7
            % Validar con geometría
            corner_ff = (form_factor >= 0.35 && form_factor <= 0.78);
            corner_sol = (solidity >= 0.65);
            corner_conv = (convexity >= 0.72);
            corner_asp = (aspect_ratio >= 0.55 && aspect_ratio <= 1.60);
            
            corner_score = corner_ff + corner_sol + corner_conv + corner_asp;
            
            if corner_score >= 3
                is_triangle_corners = true;
            end
        end
    end
    
    % ═══════════════════════════════════════════════════════════
    % MÉTODO 3: HOUGH TRANSFORM EN EL BORDE
    % ═══════════════════════════════════════════════════════════
    
    is_triangle_hough = false;
    num_lines = 0;
    num_groups = 0;
    
    if ~is_triangle_contour && ~is_triangle_corners
        % Extraer solo el borde de la región
        border = bwperim(mask_region);
        
        % Aplicar Hough
        [H, T, R] = hough(border);
        P = houghpeaks(H, 8, 'threshold', ceil(0.15*max(H(:))));
        
        if ~isempty(P)
            lines = houghlines(border, T, R, P, 'FillGap', 20, 'MinLength', 12);
            num_lines = length(lines);
            
            if num_lines >= 3 && num_lines <= 8
                % Calcular ángulos y agrupar
                angles = zeros(num_lines, 1);
                for i = 1:num_lines
                    dx = lines(i).point2(1) - lines(i).point1(1);
                    dy = lines(i).point2(2) - lines(i).point1(2);
                    angles(i) = atan2d(dy, dx);
                    if angles(i) < 0
                        angles(i) = angles(i) + 180;
                    end
                end
                
                % Agrupar líneas con ángulos similares
                angle_groups = [];
                tolerance_angle = 20;
                
                for i = 1:length(angles)
                    found = false;
                    for g = 1:length(angle_groups)
                        diff = min(abs(angles(i) - angle_groups(g)), ...
                                  abs(abs(angles(i) - angle_groups(g)) - 180));
                        if diff < tolerance_angle
                            found = true;
                            break;
                        end
                    end
                    if ~found
                        angle_groups(end+1) = angles(i);
                    end
                end
                
                num_groups = length(angle_groups);
                
                % Triángulo: 3 grupos de líneas
                if num_groups == 3
                    hough_ff = (form_factor >= 0.35 && form_factor <= 0.80);
                    hough_sol = (solidity >= 0.60);
                    hough_conv = (convexity >= 0.70);
                    
                    if hough_ff && hough_sol && hough_conv
                        is_triangle_hough = true;
                    end
                end
            end
        end
    end
    
    % ═══════════════════════════════════════════════════════════
    % DECISIÓN FINAL
    % ═══════════════════════════════════════════════════════════
    
    is_triangle = is_triangle_contour || is_triangle_corners || is_triangle_hough;
    
    % Detectar octógonos
    is_octagon = false;
    if ~is_triangle && area > 500
        octagon_ff = (form_factor > 0.77 && form_factor < 0.98);
        octagon_sol = (solidity > 0.80);
        
        if octagon_ff && octagon_sol
            % Verificar número de vértices alto
            if ~isempty(boundaries)
                boundary = boundaries{1};
                simplified = reducepoly(boundary, 0.02);
                num_vertices = size(simplified, 1);
                
                if num_vertices >= 6 && num_vertices <= 12
                    is_octagon = true;
                end
            end
        end
    end
    
    % Añadir a máscaras y reportar
    if is_triangle
        triangle_mask = triangle_mask | mask_region;
        num_triangles = num_triangles + 1;
        
        % Determinar método usado
        if is_triangle_contour
            method = sprintf('Contorno(v=%d,t=%.3f)', best_vertices, best_tolerance);
        elseif is_triangle_corners
            method = sprintf('Esquinas(n=%d)', num_corners);
        else
            method = sprintf('Hough(L=%d,G=%d)', num_lines, num_groups);
        end
        
        fprintf('  ✓ TRIÁNGULO #%d detectado (región %d):\n', num_triangles, k);
        fprintf('    - Método: %s\n', method);
        fprintf('    - Área: %d px\n', area);
        fprintf('    - Form Factor: %.3f %s\n', form_factor, tern(form_factor>=0.38 && form_factor<=0.75));
        fprintf('    - Solidez: %.3f %s\n', solidity, tern(solidity>=0.70));
        fprintf('    - Convexidad: %.3f %s\n', convexity, tern(convexity>=0.75));
        fprintf('    - Aspect: %.2f %s\n', aspect_ratio, tern(aspect_ratio>=0.60 && aspect_ratio<=1.50));
        fprintf('    - Extent: %.3f\n', extent);
        
    elseif is_octagon
        octagon_mask = octagon_mask | mask_region;
        num_octagons = num_octagons + 1;
        fprintf('  ✓ OCTÓGONO detectado (región %d)\n', k);
        
    elseif area > 800
        fprintf('  ✗ Región %d NO clasificada:\n', k);
        fprintf('    - Área=%d, FF=%.3f, Sol=%.3f, Conv=%.3f, Asp=%.2f\n', ...
            area, form_factor, solidity, convexity, aspect_ratio);
        
        % Debug: mostrar por qué no se clasificó
        if ~isempty(boundaries)
            boundary = boundaries{1};
            simplified = reducepoly(boundary, 0.02);
            fprintf('    - Vértices (t=0.02): %d\n', size(simplified, 1));
        end
    end
end

fprintf('\n✓ Resumen: %d triángulos, %d octógonos detectados\n', num_triangles, num_octagons);

subplot(3,4,7), imshow(triangle_mask), title(sprintf('7. Triángulos: %d', num_triangles));
subplot(3,4,8), imshow(octagon_mask), title(sprintf('8. Octógonos: %d', num_octagons));

% Función auxiliar
function s = tern(cond)
    if cond
        s = '✓';
    else
        s = '✗';
    end
end

%% 4. FUSIÓN INTELIGENTE - EXPANSIÓN MÍNIMA Y FIJA
fprintf('\n=== ANÁLISIS DE FUSIÓN ===\n');

shape_mask = circle_mask | triangle_mask | octagon_mask;

% Calcular métricas de solapamiento
overlap = colorMask & shape_mask;
overlap_area = sum(overlap(:));
color_area = sum(colorMask(:));
shape_area = sum(shape_mask(:));

fprintf('Área color: %d px\n', color_area);
fprintf('Área forma: %d px\n', shape_area);
fprintf('Solapamiento: %d px\n', overlap_area);

if shape_area > 0
    overlap_ratio_shape = overlap_area / shape_area;
    fprintf('Ratio solapamiento/forma: %.2f%%\n', overlap_ratio_shape * 100);
else
    overlap_ratio_shape = 0;
end

if color_area > 0
    overlap_ratio_color = overlap_area / color_area;
    fprintf('Ratio solapamiento/color: %.2f%%\n', overlap_ratio_color * 100);
else
    overlap_ratio_color = 0;
end

%% ESTRATEGIA: EXPANSIÓN MÍNIMA Y FIJA (1-2 PÍXELES)
% Expansión ultra-conservadora en todos los casos

if shape_area > 300
    fprintf('✓ Estrategia: EXPANSIÓN MÍNIMA FIJA (forma como base)\n');
    
    % PASO 1: Expansión FIJA de 1 píxel (ultra-conservador)
    expansion_size = 1; % FIJO: siempre 1 píxel
    
    fprintf('  Expansión fija: %d px\n', expansion_size);
    
    shape_expanded = imdilate(shape_mask, strel('disk', expansion_size));
    
    % PASO 2: Filtrar la expansión con la máscara de color (sin dilatar color)
    color_dilated = colorMask; % SIN DILATAR (era strel('disk', 3))
    
    % Zonas válidas: forma original + (expansión que coincide con color)
    expansion_zone = shape_expanded & ~shape_mask;
    valid_expansion = expansion_zone & color_dilated;
    
    fusion_mask = shape_mask | valid_expansion;
    
    fprintf('  Área forma original: %d px\n', sum(shape_mask(:)));
    fprintf('  Área expansión válida: %d px\n', sum(valid_expansion(:)));
    fprintf('  Área total fusión: %d px\n', sum(fusion_mask(:)));
    
    % PASO 3: Rellenar huecos internos
    fusion_mask = imfill(fusion_mask, 'holes');
    
    % PASO 4: Suavizar bordes MÍNIMO
    fusion_mask = imclose(fusion_mask, strel('disk', 1)); % Era 3, ahora 1
    
elseif shape_area > 100
    fprintf('✓ Estrategia: FORMA PEQUEÑA + COLOR (expansión reducida)\n');
    shape_expanded = imdilate(shape_mask, strel('disk', 2)); % Era 10, ahora 2
    color_close = imclose(colorMask, strel('disk', 2)); % Era 5, ahora 2
    fusion_mask = shape_expanded & color_close;
    fusion_mask = imfill(fusion_mask, 'holes');
    
elseif color_area > 400
    fprintf('✓ Estrategia: COLOR COMO BASE (expansión reducida)\n');
    fusion_mask = imclose(colorMask, strel('disk', 2)); % Era 8, ahora 2
    fusion_mask = imfill(fusion_mask, 'holes');
    fusion_mask = bwareaopen(fusion_mask, 300);
    
else
    fprintf('✓ Estrategia: RESCATE (expansión mínima)\n');
    fusion_mask = imdilate(colorMask, strel('disk', 2)) | imdilate(shape_mask, strel('disk', 2)); % Era 8 y 12, ahora 2 y 2
    fusion_mask = imfill(fusion_mask, 'holes');
end

subplot(3,4,9), imshow(fusion_mask), title('9. Fusión Mínima');

%% 5. REFINAMIENTO CON CONTEXTO GEOMÉTRICO
% Si detectamos círculos, asegurar que la máscara los incluya (EXPANSIÓN MÍNIMA)
if num_circles > 0
    fprintf('Refinando con contexto de círculos...\n');
    for i = 1:num_circles
        [xx, yy] = meshgrid(1:cols, 1:rows);
        % Crear círculo con EXPANSIÓN MÍNIMA
        circle_complete = ((xx - circle_centers(i,1)).^2 + (yy - circle_centers(i,2)).^2) <= (circle_radii(i)*1.02)^2; % Era 1.15 (15%), ahora 1.02 (2%)
        
        % Si hay algo de overlap con fusion_mask, incluir todo el círculo
        if sum(fusion_mask(:) & circle_complete(:)) > 100
            fusion_mask = fusion_mask | circle_complete;
        end
    end
end

subplot(3,4,10), imshow(fusion_mask), title('10. Con Contexto Geométrico');

%% 6. MORFOLOGÍA FINAL - MÍNIMA
se1 = strel('disk', 1); % Era 2, ahora 1
se2 = strel('disk', 2); % Era 4, ahora 2

final_mask = imclose(fusion_mask, se2);
final_mask = imfill(final_mask, 'holes');
final_mask = imopen(final_mask, se1);
final_mask = bwareaopen(final_mask, 200);

% Morfología final suave (reducida)
final_mask = imclose(final_mask, strel('disk', 2)); % Era 5, ahora 2
final_mask = imfill(final_mask, 'holes');

subplot(3,4,11), imshow(final_mask), title('11. Morfología Final');

%% 7. SELECCIÓN DE MEJOR REGIÓN
stats = regionprops(final_mask, 'Area', 'BoundingBox', 'Solidity', ...
    'PixelIdxList', 'Eccentricity', 'Centroid', 'Perimeter', 'Extent');

if isempty(stats)
    fprintf('\n⚠️  No se detectaron regiones finales\n');
    img_final = img;
    mascara_final = final_mask;
else
    valid = false(length(stats), 1);
    scores = zeros(length(stats), 1);
    shape_types = cell(length(stats), 1);
    
    for i = 1:length(stats)
        area = stats(i).Area;
        bbox = stats(i).BoundingBox;
        aspect = bbox(3) / bbox(4);
        perimeter = stats(i).Perimeter;
        
        % Criterios de validación más permisivos
        area_ok = area > 300 && area < rows*cols*0.8;
        solidity_ok = stats(i).Solidity > 0.55;
        aspect_ok = aspect > 0.4 && aspect < 2.5;
        extent_ok = stats(i).Extent > 0.3;
        
        valid(i) = area_ok && solidity_ok && aspect_ok && extent_ok;
        
        if valid(i)
            form_factor = (4 * pi * area) / (perimeter^2);
            
            % Clasificar forma
            if form_factor > 0.82
                if stats(i).Solidity > 0.88
                    shape_types{i} = 'Círculo/Octógono';
                else
                    shape_types{i} = 'Círculo';
                end
            elseif form_factor > 0.4 && form_factor < 0.72
                shape_types{i} = 'Triángulo';
            else
                shape_types{i} = 'Señal Irregular';
            end
            
            % Score mejorado: área + solidez + centralidad + compacidad
            center_img = [cols/2, rows/2];
            dist_center = norm(stats(i).Centroid - center_img);
            normalized_dist = dist_center / norm(center_img);
            
            % Penalizar menos la distancia, priorizar área y solidez
            scores(i) = area * stats(i).Solidity * stats(i).Extent * (1.3 - normalized_dist*0.5);
        end
    end
    
    stats = stats(valid);
    scores = scores(valid);
    shape_types = shape_types(valid);
    
    if isempty(stats)
        fprintf('\n⚠️  No hay regiones válidas después del filtrado\n');
        fprintf('Intentando con criterios más permisivos...\n');
        
        % Segundo intento con criterios ultra-permisivos
        stats_all = regionprops(final_mask, 'Area', 'BoundingBox', 'Solidity', ...
            'PixelIdxList', 'Centroid', 'Perimeter');
        
        if ~isempty(stats_all)
            % Tomar la región más grande
            [~, idx_max] = max([stats_all.Area]);
            best = stats_all(idx_max);
            detected_shape = 'Señal (detección permisiva)';
            
            mascara_final = false(rows, cols);
            mascara_final(best.PixelIdxList) = true;
            mascara_final = imclose(mascara_final, strel('disk', 2)); % Era 5, ahora 2
            mascara_final = imfill(mascara_final, 'holes');
            
            fprintf('✓ Usando región más grande con criterios permisivos\n');
        else
            img_final = img;
            mascara_final = final_mask;
        end
    else
        [~, idx] = max(scores);
        best = stats(idx);
        detected_shape = shape_types{idx};
        
        mascara_final = false(rows, cols);
        mascara_final(best.PixelIdxList) = true;
        
        % Suavizar bordes finales - MÍNIMO
        mascara_final = imclose(mascara_final, strel('disk', 2)); % Era 4, ahora 2
        mascara_final = imfill(mascara_final, 'holes');
    end
    
    if exist('best', 'var')
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
            
            % Visualizar con bbox
            img_bbox = insertShape(img, 'Rectangle', bbox, ...
                'Color', 'green', 'LineWidth', 3);
            
            position = [bbox(1)+5, bbox(2)-15];
            img_bbox = insertText(img_bbox, position, detected_shape, ...
                'FontSize', 14, 'BoxColor', 'green', 'TextColor', 'white');
            
            subplot(3,4,12), imshow(img_bbox), title('12. Detección Final');
            
            fprintf('\n✅ SEÑAL DETECTADA EXITOSAMENTE\n');
            fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
            fprintf('Tipo detectado: %s\n', detected_shape);
            fprintf('Área: %d px (%.1f%% de la imagen)\n', best.Area, best.Area/(rows*cols)*100);
            fprintf('Solidez: %.2f\n', best.Solidity);
            fprintf('Perímetro: %.1f px\n', best.Perimeter);
            fprintf('Factor de forma: %.3f\n', (4*pi*best.Area)/(best.Perimeter^2));
            fprintf('BoundingBox: [%d, %d, %d×%d]\n', x, y, w, h);
            fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
        end
    end
end

%% 8. RESUMEN VISUAL FINAL
figure('Position', [100, 100, 1600, 400]);
subplot(1,5,1), imshow(img), title('Original', 'FontSize', 12);
subplot(1,5,2), imshow(colorMask), title('Máscara Color', 'FontSize', 12);
subplot(1,5,3), imshow(shape_mask), title(sprintf('Formas: C=%d T=%d O=%d', ...
    num_circles, num_triangles, num_octagons), 'FontSize', 12);
subplot(1,5,4), imshow(mascara_final), title('Máscara Final', 'FontSize', 12);
subplot(1,5,5), imshow(img_final), title('Segmentación', 'FontSize', 12);

sgtitle('Pipeline Completo de Detección - Expansión Mínima', 'FontSize', 14, 'FontWeight', 'bold');