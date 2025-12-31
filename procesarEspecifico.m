%% SISTEMA ROBUSTO DE DETECCIÓN DE SEÑALES - MÁSCARAS POR COLOR
% Separa detección por color: Rojo, Azul, Amarillo
clear; clc; close all;

train_path = 'imatges_senyals/test';
categoria = 'zona_bici';
nombre_archivo = '030_0074.png';

%% Cargar imagen
img_path = fullfile(train_path, categoria, nombre_archivo);
if ~exist(img_path, 'file')
    fprintf('ERROR: No se encuentra la imagen: %s\n', img_path);
    return;
end

img = imread(img_path);
[rows, cols, ~] = size(img);

figure('Position', [50, 50, 1600, 900]);
subplot(3,5,1), imshow(img), title('1. Original');

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

subplot(3,5,2), imshow(rgb_norm), title('2. Normalizado');

%% 2. DETECCIÓN POR COLOR - MÁSCARAS SEPARADAS

% ===== MÁSCARA ROJA =====
red_rgb = (R > 0.5) & (G < 0.5) & (B < 0.5);
red_hsv = (H > 0.94 | H < 0.06) & (S > 0.4) & (V > 0.3);
red_mask = (red_rgb | red_hsv) & (S > 0.3);

% ===== MÁSCARA AZUL =====
blue_rgb = (B > 0.5) & (R < 0.4) & (G < 0.5);
blue_hsv = (H > 0.54 & H < 0.70) & (S > 0.35) & (V > 0.25);
blue_mask_raw = (blue_rgb | blue_hsv) & (S > 0.3);

% FILTRAR: Eliminar regiones grandes de fondo (cielo)
blue_mask = filtrar_fondo_grande(blue_mask_raw, rows, cols);

% ===== MÁSCARA AMARILLA =====
yellow_rgb = (R > 0.5) & (G > 0.45) & (B < 0.35);
yellow_hsv = (H > 0.10 & H < 0.20) & (S > 0.4) & (V > 0.35);
orange_rgb = (R > 0.7) & (G > 0.3 & G < 0.7) & (B < 0.35);
orange_hsv = (H > 0.04 & H < 0.11) & (S > 0.4) & (V > 0.35);
yellow_mask_raw = (yellow_rgb | yellow_hsv | orange_rgb | orange_hsv) & (S > 0.3);

% FILTRAR: Eliminar regiones grandes de fondo
yellow_mask = filtrar_fondo_grande(yellow_mask_raw, rows, cols);

fprintf('\n=== MÁSCARAS DE COLOR DETECTADAS ===\n');
fprintf('Área ROJA: %d px\n', sum(red_mask(:)));
fprintf('Área AZUL (antes filtrado): %d px\n', sum(blue_mask_raw(:)));
fprintf('Área AZUL (después filtrado): %d px\n', sum(blue_mask(:)));
fprintf('Área AMARILLA: %d px\n', sum(yellow_mask(:)));

subplot(3,5,3), imshow(red_mask), title('3. Máscara ROJA');
subplot(3,5,4), imshow(blue_mask), title('4. Máscara AZUL');
subplot(3,5,5), imshow(yellow_mask), title('5. Máscara AMARILLA');

%% 3. DETECCIÓN DE BORDES (común para todas)
img_enhanced = imadjust(img_gray);
edges_all = edge(img_enhanced, 'Canny', [0.08 0.22]);

subplot(3,5,6), imshow(edges_all), title('6. Edges Canny');

%% 4. DETECCIÓN DE FORMAS POR COLOR

% Inicializar contadores y máscaras
num_circles_red = 0;
num_circles_blue = 0;
num_triangles_yellow = 0;
num_octagons_red = 0;

circle_mask_red = false(rows, cols);
circle_mask_blue = false(rows, cols);
triangle_mask_yellow = false(rows, cols);
octagon_mask_red = false(rows, cols);
% ═══════════════════════════════════════════════════════════
% A) CÍRCULOS ROJOS (Stop completo + Prohibición con borde)
% ═══════════════════════════════════════════════════════════
fprintf('\n=== DETECCIÓN DE CÍRCULOS ROJOS ===\n');

radio_min = 15;
radio_max = round(min(rows, cols) / 2);

if sum(red_mask(:)) > 300
    fprintf('Intentando método mejorado (perímetro)...\n');
    
    red_mask_clean = imclose(red_mask, strel('disk', 5));
    red_mask_clean = imfill(red_mask_clean, 'holes');
    red_mask_clean = bwareaopen(red_mask_clean, 300);
    
    CC = bwconncomp(red_mask_clean);
    
    % **VERIFICACIÓN: ¿Hay componentes conectados?**
    if CC.NumObjects == 0
        fprintf('⚠ No hay componentes conectados después de limpiar la máscara roja\n');
        
    else
        % Continúa con el método mejorado
        numPixels = cellfun(@numel, CC.PixelIdxList);
        [~, idx_largest] = max(numPixels);
        
        red_mask_largest = false(rows, cols);
        red_mask_largest(CC.PixelIdxList{idx_largest}) = true;
        
        red_boundary = bwperim(red_mask_largest);
        red_boundary = imdilate(red_boundary, strel('disk', 3));
        
        img_red_edge = img_enhanced;
        img_red_edge(~red_boundary) = 255;
        
        [centers, radii, metric] = imfindcircles(img_red_edge, [radio_min radio_max], ...
            'ObjectPolarity', 'dark', ...
            'Sensitivity', 0.97, ...
            'EdgeThreshold', 0.02, ...
            'Method', 'TwoStage');
        
        all_centers = centers;
        all_radii = radii;
        
        if ~isempty(centers)
            % Éxito con método mejorado
            fprintf('✓ Método mejorado: %d círculos encontrados\n', length(radii));
            
            [max_radio, best_idx] = max(radii);
            fprintf('  Círculo rojo seleccionado: Radio=%.1f (el más grande)\n', max_radio);
            
            centers = centers(best_idx, :);
            radii = radii(best_idx);
            num_circles_red = 1;
            
            [xx, yy] = meshgrid(1:cols, 1:rows);
            circle_temp = ((xx - centers(1)).^2 + (yy - centers(2)).^2) <= (radii*1.02)^2;
            circle_mask_red = circle_temp;
            
        else
            % Método mejorado no encontró círculos
            fprintf('⚠ Método mejorado no detectó círculos rojos\n');
        end
    end
end

subplot(3,5,7);
if num_circles_red > 0
    imshow(img);
    if ~isempty(all_centers) && size(all_centers, 1) > 1
        % Mostrar todos los candidatos en rojo
        viscircles(all_centers, all_radii, 'Color', 'r', 'LineWidth', 1.5);
    end
    % Mostrar el seleccionado en verde
    viscircles(centers, radii, 'Color', 'g', 'LineWidth', 2);
    title(sprintf('7. Círculos ROJOS: %d', num_circles_red));
else
    imshow(img), title('7. Círculos ROJOS: 0');
end

% ═══════════════════════════════════════════════════════════
% B) CÍRCULOS AZULES (Obligación) - MEJORADO
% ═══════════════════════════════════════════════════════════
fprintf('\n=== DETECCIÓN DE CÍRCULOS AZULES ===\n');

if sum(blue_mask(:)) > 300
    % PASO 1: Limpiar y obtener región más grande
    blue_mask_clean = imclose(blue_mask, strel('disk', 5));
    blue_mask_clean = imfill(blue_mask_clean, 'holes');
    blue_mask_clean = bwareaopen(blue_mask_clean, 300);
    
    CC = bwconncomp(blue_mask_clean);
    numPixels = cellfun(@numel, CC.PixelIdxList);
    [~, idx_largest] = max(numPixels);
    
    blue_mask_largest = false(rows, cols);
    blue_mask_largest(CC.PixelIdxList{idx_largest}) = true;
    
    % PASO 2: Obtener el BORDE de la región más grande
    blue_boundary = bwperim(blue_mask_largest);
    blue_boundary = imdilate(blue_boundary, strel('disk', 3)); % Engrosar más
    
    % PASO 3: Aplicar a imagen
    img_blue_edge = img_enhanced;
    img_blue_edge(~blue_boundary) = 255; % Fondo blanco
    
    radio_min = 20; % Menos restrictivo
    
    [centers_blue, radii_blue, metric_blue] = imfindcircles(img_blue_edge, [radio_min radio_max], ...
        'ObjectPolarity', 'dark', ...
        'Sensitivity', 0.97, ...      % ↑ Más sensible
        'EdgeThreshold', 0.02, ...    % ↓ Menos restrictivo
        'Method', 'TwoStage');        % Más robusto
    
    if ~isempty(centers_blue)
        fprintf('Círculos azules encontrados: %d\n', length(radii_blue));
        
        % Filtrar: tomar solo el MÁS GRANDE
        [max_radio, idx_best] = max(radii_blue);
        
        fprintf('  ✓ Círculo azul seleccionado: Radio=%.1f (el más grande)\n', max_radio);
        
        centers_blue = centers_blue(idx_best, :);
        radii_blue = radii_blue(idx_best);
        
        num_circles_blue = 1;
        [xx, yy] = meshgrid(1:cols, 1:rows);
        circle_temp = ((xx - centers_blue(1)).^2 + (yy - centers_blue(2)).^2) <= (radii_blue*1.02)^2;
        circle_mask_blue = circle_temp;
    else
        fprintf('  ✗ No se detectaron círculos azules\n');
    end
end

subplot(3,5,8);
if num_circles_blue > 0
    imshow(img);
    viscircles(centers_blue(1,:), radii_blue(1), 'Color', 'b', 'LineWidth', 2);
    title(sprintf('8. Círculos AZULES: %d', num_circles_blue));
else
    imshow(img), title('8. Círculos AZULES: 0');
end

% ═══════════════════════════════════════════════════════════
% C) BÚSQUEDA DE RESPALDO: Círculos en imagen original
%    (Solo si no se encontraron círculos rojos ni azules)
% ═══════════════════════════════════════════════════════════
if num_circles_red == 0 && num_circles_blue == 0
    fprintf('\n=== BÚSQUEDA DE RESPALDO: CÍRCULOS EN IMAGEN ORIGINAL ===\n');
    
    [centers, radii, metric] = imfindcircles(img_enhanced, [radio_min radio_max], ...
        'ObjectPolarity', 'dark', 'Sensitivity', 0.93, 'EdgeThreshold', 0.08);
    
    if ~isempty(centers)
        fprintf('✓ Círculos encontrados en imagen original: %d\n', length(radii));
        
        % Filtrar círculos con scoring
        if length(radii) > 1
            center_img = [cols/2, rows/2];
            circle_scores = zeros(length(radii), 1);
            
            for i = 1:length(radii)
                [xx, yy] = meshgrid(1:cols, 1:rows);
                circle_temp = ((xx - centers(i,1)).^2 + (yy - centers(i,2)).^2) <= radii(i)^2;
                
                dist_to_center = norm([centers(i,1), centers(i,2)] - center_img);
                max_dist = norm(center_img);
                centrality = 1 - (dist_to_center / max_dist);
                
                color_overlap = sum(red_mask(:) & circle_temp(:)) / sum(circle_temp(:));
                
                circle_scores(i) = metric(i) * 0.3 + ...
                                  (radii(i)/radio_max) * 0.2 + ...
                                  centrality * 0.2 + ...
                                  color_overlap * 0.3;
                
                fprintf('  Círculo %d: Radio=%.1f, ColorOverlap=%.2f, Score=%.3f\n', ...
                    i, radii(i), color_overlap, circle_scores(i));
            end
            
            [~, best_idx] = max(circle_scores);
            fprintf('✓ Seleccionado círculo #%d (respaldo)\n', best_idx);
            
            centers = centers(best_idx, :);
            radii = radii(best_idx);
        end
        
        num_circles_red = length(radii);
        
        for i = 1:num_circles_red
            [xx, yy] = meshgrid(1:cols, 1:rows);
            circle_temp = ((xx - centers(i,1)).^2 + (yy - centers(i,2)).^2) <= (radii(i)*1.02)^2;
            circle_mask_red = circle_mask_red | circle_temp;
        end
        
        % Actualizar visualización del subplot 7
        subplot(3,5,7);
        imshow(img);
        viscircles(centers, radii, 'Color', 'm', 'LineWidth', 2); % Magenta para indicar respaldo
        title(sprintf('7. Círculos ROJOS: %d (respaldo)', num_circles_red));
    else
        fprintf('✗ No se detectaron círculos con búsqueda de respaldo\n');
    end
end

% ═══════════════════════════════════════════════════════════
% B) CÍRCULOS AZULES (Obligación) - MEJORADO
% ═══════════════════════════════════════════════════════════
fprintf('\n=== DETECCIÓN DE CÍRCULOS AZULES ===\n');

if sum(blue_mask(:)) > 300
    % PASO 1: Limpiar y obtener región más grande
    blue_mask_clean = imclose(blue_mask, strel('disk', 5));
    blue_mask_clean = imfill(blue_mask_clean, 'holes');
    blue_mask_clean = bwareaopen(blue_mask_clean, 300);
    
    CC = bwconncomp(blue_mask_clean);
    numPixels = cellfun(@numel, CC.PixelIdxList);
    [~, idx_largest] = max(numPixels);
    
    blue_mask_largest = false(rows, cols);
    blue_mask_largest(CC.PixelIdxList{idx_largest}) = true;
    
    % PASO 2: Obtener el BORDE de la región más grande
    blue_boundary = bwperim(blue_mask_largest);
    blue_boundary = imdilate(blue_boundary, strel('disk', 3)); % Engrosar más
    
    % PASO 3: Aplicar a imagen
    img_blue_edge = img_enhanced;
    img_blue_edge(~blue_boundary) = 255; % Fondo blanco
    
    radio_min = 20; % Menos restrictivo
    
    [centers_blue, radii_blue, metric_blue] = imfindcircles(img_blue_edge, [radio_min radio_max], ...
        'ObjectPolarity', 'dark', ...
        'Sensitivity', 0.97, ...      % ↑ Más sensible
        'EdgeThreshold', 0.02, ...    % ↓ Menos restrictivo
        'Method', 'TwoStage');        % Más robusto
    
    if ~isempty(centers_blue)
        fprintf('Círculos azules encontrados: %d\n', length(radii_blue));
        
        % Filtrar: tomar solo el MÁS GRANDE
        [max_radio, idx_best] = max(radii_blue);
        
        fprintf('  ✓ Círculo azul seleccionado: Radio=%.1f (el más grande)\n', max_radio);
        
        centers_blue = centers_blue(idx_best, :);
        radii_blue = radii_blue(idx_best);
        
        num_circles_blue = 1;
        [xx, yy] = meshgrid(1:cols, 1:rows);
        circle_temp = ((xx - centers_blue(1)).^2 + (yy - centers_blue(2)).^2) <= (radii_blue*1.02)^2;
        circle_mask_blue = circle_temp;
    else
        fprintf('  ✗ No se detectaron círculos azules\n');
    end
end

subplot(3,5,8);
if num_circles_blue > 0
    imshow(img);
    viscircles(centers_blue(1,:), radii_blue(1), 'Color', 'b', 'LineWidth', 2);
    title(sprintf('8. Círculos AZULES: %d', num_circles_blue));
else
    imshow(img), title('8. Círculos AZULES: 0');
end
% ═══════════════════════════════════════════════════════════
% C) TRIÁNGULOS AMARILLOS (Advertencia) - DETECCIÓN MEJORADA
% ═══════════════════════════════════════════════════════════
fprintf('\n=== DETECCIÓN DE TRIÁNGULOS AMARILLOS ===\n');

if sum(yellow_mask(:)) > 300
    % PASO 1: Limpieza agresiva para cerrar huecos del símbolo interno
    yellow_mask_clean = imclose(yellow_mask, strel('disk', 4)); % Aumentado de 4 a 8
    yellow_mask_clean = imfill(yellow_mask_clean, 'holes'); % Rellenar TODO
    
    % PASO 2: Dilatación adicional para unir fragmentos del borde
    yellow_mask_dilated = imdilate(yellow_mask_clean, strel('disk', 3));
    yellow_mask_dilated = imfill(yellow_mask_dilated, 'holes');
    
    % PASO 3: Erosión para volver al tamaño aproximado original
    yellow_mask_final = imerode(yellow_mask_dilated, strel('disk', 3));
    yellow_mask_final = bwareaopen(yellow_mask_final, 300);
    
    % También usar edges pero con máscara amarilla dilatada
    yellow_mask_edges = imdilate(yellow_mask, strel('disk', 3)); % Más amplio
    edges_yellow = edges_all & yellow_mask_edges;
    
    % Combinar edges cerrados
    edges_yellow_clean = imclose(edges_yellow, strel('disk', 3));
    edges_yellow_clean = imfill(edges_yellow_clean, 'holes');
    edges_yellow_clean = bwareaopen(edges_yellow_clean, 300);
    
    % ESTRATEGIA DUAL: analizar regiones de ambas fuentes
    CC_mask = bwconncomp(yellow_mask_final);
    CC_edges = bwconncomp(edges_yellow_clean);
    
    fprintf('Candidatos desde máscara amarilla: %d\n', CC_mask.NumObjects);
    fprintf('Candidatos desde edges amarillos: %d\n', CC_edges.NumObjects);
    
    % Combinar todas las regiones candidatas
    all_yellow_regions = {};
    for k = 1:CC_mask.NumObjects
        mask_region = false(rows, cols);
        mask_region(CC_mask.PixelIdxList{k}) = true;
        all_yellow_regions{end+1} = mask_region;
    end
    for k = 1:CC_edges.NumObjects
        mask_region = false(rows, cols);
        mask_region(CC_edges.PixelIdxList{k}) = true;
        all_yellow_regions{end+1} = mask_region;
    end
    
    % Analizar cada región candidata
    for idx = 1:length(all_yellow_regions)
        mask_region = all_yellow_regions{idx};
        
        stats = regionprops(mask_region, 'Area', 'Perimeter', 'BoundingBox', ...
            'Solidity', 'Extent', 'ConvexArea', 'Eccentricity', 'Centroid', ...
            'MajorAxisLength', 'MinorAxisLength');
        
        if isempty(stats)
            continue;
        end
        
        area = stats(1).Area;
        perimeter = stats(1).Perimeter;
        bbox = stats(1).BoundingBox;
        
        % Filtros básicos
        if area < 300 || area > rows*cols*0.75 || bbox(3) < 15 || bbox(4) < 15
            continue;
        end
        
        solidity = stats(1).Solidity;
        convex_area = stats(1).ConvexArea;
        extent = stats(1).Extent;
        
        if perimeter > 0
            form_factor = (4 * pi * area) / (perimeter^2);
        else
            continue;
        end
        
        convexity = area / convex_area;
        aspect_ratio = bbox(3) / bbox(4);
        
        % Verificar overlap con máscara amarilla original
        overlap_yellow = sum(yellow_mask(:) & mask_region(:)) / area;
        
        if overlap_yellow < 0.3 % Debe tener algo de amarillo
            continue;
        end
        
        % ═══════════════════════════════════════════════════════════
        % DETECCIÓN DE TRIÁNGULOS: Métodos robustos
        % ═══════════════════════════════════════════════════════════
        is_triangle = false;
        detection_method = '';
        
        % MÉTODO 1: Análisis del contorno EXTERNO (ignorando huecos internos)
        % Este es el método más robusto para triángulos con símbolos dentro
        boundaries = bwboundaries(mask_region);
        
        if ~isempty(boundaries)
            % Tomar solo el contorno externo (el primero es siempre el más grande)
            boundary = boundaries{1};
            
            % Probar múltiples tolerancias para simplificación
            for tolerance = [0.005, 0.008, 0.01, 0.012, 0.015, 0.018, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045]
                simplified = reducepoly(boundary, tolerance);
                num_vertices = size(simplified, 1);
                
                % Triángulos: 3-7 vértices (muy permisivo para compensar irregularidades)
                if num_vertices >= 3 && num_vertices <= 7
                    % Criterios RELAJADOS para triángulos amarillos
                    ff_ok = (form_factor >= 0.30 && form_factor <= 0.78);  % Más permisivo
                    sol_ok = (solidity >= 0.65);  % Más permisivo
                    conv_ok = (convexity >= 0.65); % Más permisivo
                    asp_ok = (aspect_ratio >= 0.55 && aspect_ratio <= 1.60); % Más rango
                    ext_ok = (extent >= 0.35); % Más permisivo
                    
                    score = ff_ok + sol_ok + conv_ok + asp_ok + ext_ok;
                    
                    % Con 3 de 5 criterios es suficiente
                    if score >= 3
                        is_triangle = true;
                        detection_method = sprintf('Contorno(%d vértices, t=%.3f)', num_vertices, tolerance);
                        break;
                    end
                end
            end
        end
        
        % MÉTODO 2: Análisis del casco convexo (forma exterior ideal)
        if ~is_triangle
            % Obtener el casco convexo de la región
            [B_y, B_x] = find(mask_region);
            if length(B_x) > 10
                try
                    k = convhull(B_x, B_y);
                    convex_boundary = [B_x(k), B_y(k)];
                    
                    % Simplificar el casco convexo
                    for tolerance = [0.01, 0.015, 0.02, 0.025, 0.03]
                        simplified_convex = reducepoly(convex_boundary, tolerance);
                        num_vertices_convex = size(simplified_convex, 1);
                        
                        if num_vertices_convex >= 3 && num_vertices_convex <= 6
                            % El casco convexo tiene forma triangular
                            if form_factor >= 0.30 && solidity >= 0.60 && convexity >= 0.60
                                is_triangle = true;
                                detection_method = sprintf('CascoConvexo(%d vértices)', num_vertices_convex);
                                break;
                            end
                        end
                    end
                catch
                    % Ignorar errores de convhull
                end
            end
        end
        
        % MÉTODO 3: Detección por esquinas en el perímetro
        if ~is_triangle && area > 400
            % Aplicar detección de esquinas sobre la máscara
            mask_uint8 = uint8(mask_region) * 255;
            corners = detectHarrisFeatures(mask_uint8, 'MinQuality', 0.01);
            
            if corners.Count >= 3 && corners.Count <= 9
                % Filtrar esquinas que estén cerca del perímetro
                boundary_mask = bwperim(mask_region);
                boundary_dilated = imdilate(boundary_mask, strel('disk', 5));
                
                corners_on_boundary = 0;
                for c = 1:corners.Count
                    cx = round(corners.Location(c, 1));
                    cy = round(corners.Location(c, 2));
                    if cx > 0 && cx <= cols && cy > 0 && cy <= rows
                        if boundary_dilated(cy, cx)
                            corners_on_boundary = corners_on_boundary + 1;
                        end
                    end
                end
                
                % Si hay 3-5 esquinas en el perímetro, probablemente es un triángulo
                if corners_on_boundary >= 3 && corners_on_boundary <= 6
                    if form_factor >= 0.28 && solidity >= 0.60 && convexity >= 0.60
                        is_triangle = true;
                        detection_method = sprintf('Esquinas(%d en perímetro)', corners_on_boundary);
                    end
                end
            end
        end
        
        % MÉTODO 4: Criterios geométricos puros (forma general triangular)
        if ~is_triangle
            % Un triángulo tiene características específicas:
            % - Form factor entre 0.3-0.7 (más bajo que círculo)
            % - Solidez moderada (0.65-0.92)
            % - Excentricidad no muy alta (no es una línea)
            
            eccentricity = stats(1).Eccentricity;
            
            is_triangular_shape = (form_factor >= 0.32 && form_factor <= 0.72) && ...
                                  (solidity >= 0.65 && solidity <= 0.94) && ...
                                  (convexity >= 0.65) && ...
                                  (eccentricity >= 0.3 && eccentricity <= 0.95) && ...
                                  (aspect_ratio >= 0.60 && aspect_ratio <= 1.50) && ...
                                  (extent >= 0.38);
            
            if is_triangular_shape
                is_triangle = true;
                detection_method = 'Geometría triangular pura';
            end
        end
        
        % REGISTRAR RESULTADO
        if is_triangle
            triangle_mask_yellow = triangle_mask_yellow | mask_region;
            num_triangles_yellow = num_triangles_yellow + 1;
            fprintf('  ✓ TRIÁNGULO AMARILLO #%d detectado:\n', num_triangles_yellow);
            fprintf('    - Método: %s\n', detection_method);
            fprintf('    - Área: %d px\n', area);
            fprintf('    - Form Factor: %.3f\n', form_factor);
            fprintf('    - Solidez: %.3f\n', solidity);
            fprintf('    - Convexidad: %.3f\n', convexity);
            fprintf('    - Aspect Ratio: %.2f\n', aspect_ratio);
            fprintf('    - Extent: %.3f\n', extent);
            fprintf('    - Overlap amarillo: %.2f%%\n', overlap_yellow*100);
        elseif area > 500 && overlap_yellow > 0.4
            % Debug: mostrar por qué no se detectó
            fprintf('  ✗ Región amarilla #%d NO clasificada:\n', idx);
            fprintf('    - Área=%d, FF=%.3f, Sol=%.3f, Conv=%.3f\n', ...
                area, form_factor, solidity, convexity);
            fprintf('    - Aspect=%.2f, Extent=%.3f, Overlap=%.2f%%\n', ...
                aspect_ratio, extent, overlap_yellow*100);
        end
    end
    
    fprintf('✓ Total triángulos amarillos: %d\n', num_triangles_yellow);
end

subplot(3,5,9), imshow(triangle_mask_yellow), title(sprintf('9. Triángulos AMARILLOS: %d', num_triangles_yellow));

% ═══════════════════════════════════════════════════════════
% D) OCTÓGONOS ROJOS (Stop) - DETECCIÓN MEJORADA
% ═══════════════════════════════════════════════════════════
fprintf('\n=== DETECCIÓN DE OCTÓGONOS ROJOS ===\n');

if sum(red_mask(:)) > 500
    % ESTRATEGIA DUAL: Usar máscara roja limpia Y edges
    
    % Método A: Desde máscara roja con limpieza mínima
    red_mask_clean = imclose(red_mask, strel('disk', 2)); % Mínimo suavizado
    red_mask_clean = imfill(red_mask_clean, 'holes');
    red_mask_clean = bwareaopen(red_mask_clean, 300);
    
    % Método B: Desde edges en zona roja
    red_mask_dilated = imdilate(red_mask, strel('disk', 5));
    edges_red = edges_all & red_mask_dilated;
    edges_red_clean = imclose(edges_red, strel('disk', 2));
    edges_red_clean = imfill(edges_red_clean, 'holes');
    edges_red_clean = bwareaopen(edges_red_clean, 300);
    
    % Combinar ambas fuentes
    CC_red_mask = bwconncomp(red_mask_clean);
    CC_red_edges = bwconncomp(edges_red_clean);
    
    fprintf('Candidatos desde máscara roja: %d\n', CC_red_mask.NumObjects);
    fprintf('Candidatos desde edges rojos: %d\n', CC_red_edges.NumObjects);
    
    % Analizar ambas fuentes
    all_regions = {};
    for k = 1:CC_red_mask.NumObjects
        mask_region = false(rows, cols);
        mask_region(CC_red_mask.PixelIdxList{k}) = true;
        all_regions{end+1} = mask_region;
    end
    for k = 1:CC_red_edges.NumObjects
        mask_region = false(rows, cols);
        mask_region(CC_red_edges.PixelIdxList{k}) = true;
        all_regions{end+1} = mask_region;
    end
    
    for idx = 1:length(all_regions)
        mask_region = all_regions{idx};
        
        stats = regionprops(mask_region, 'Area', 'Perimeter', 'Solidity', ...
            'BoundingBox', 'ConvexArea', 'Extent', 'Eccentricity');
        
        if isempty(stats)
            continue;
        end
        
        area = stats(1).Area;
        perimeter = stats(1).Perimeter;
        bbox = stats(1).BoundingBox;
        
        % Filtros básicos más permisivos
        if area < 300 || area > rows*cols*0.75 || perimeter == 0
            continue;
        end
        
        if bbox(3) < 15 || bbox(4) < 15
            continue;
        end
        
        form_factor = (4 * pi * area) / (perimeter^2);
        solidity = stats(1).Solidity;
        convexity = area / stats(1).ConvexArea;
        aspect_ratio = bbox(3) / bbox(4);
        extent = stats(1).Extent;
        
        % Verificar overlap con máscara roja
        overlap_red = sum(red_mask(:) & mask_region(:)) / area;
        
        if overlap_red < 0.4 % Más permisivo
            continue;
        end
        
        % DETECCIÓN DE OCTÓGONOS: Múltiples métodos
        is_octagon = false;
        detection_method = '';
        
        % MÉTODO 1: Análisis de vértices en contorno
        boundaries = bwboundaries(mask_region, 'noholes');
        best_vertices = 0;
        
        if ~isempty(boundaries)
            boundary = boundaries{1};
            
            % Probar varias tolerancias
            for tolerance = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
                simplified = reducepoly(boundary, tolerance);
                num_vertices = size(simplified, 1);
                
                if num_vertices > best_vertices
                    best_vertices = num_vertices;
                end
                
                % Octógonos: 6-14 vértices (MUY permisivo)
                if num_vertices >= 6 && num_vertices <= 14
                    % Criterios geométricos RELAJADOS
                    ff_ok = (form_factor > 0.70 && form_factor < 0.99);
                    sol_ok = (solidity > 0.70);
                    conv_ok = (convexity > 0.75);
                    asp_ok = (aspect_ratio > 0.70 && aspect_ratio < 1.45);
                    ext_ok = (extent > 0.40);
                    
                    score = ff_ok + sol_ok + conv_ok + asp_ok + ext_ok;
                    
                    if score >= 3
                        is_octagon = true;
                        detection_method = sprintf('Vértices(%d, t=%.3f)', num_vertices, tolerance);
                        break;
                    end
                end
            end
        end
        
        % MÉTODO 2: Form Factor alto (casi circular pero no perfecto)
        if ~is_octagon
            if form_factor > 0.75 && form_factor < 0.96
                if solidity > 0.80 && convexity > 0.80 && extent > 0.45
                    % Verificar que no sea un círculo detectado
                    if num_circles_red == 0 || area > sum(circle_mask_red(:)) * 0.5
                        is_octagon = true;
                        detection_method = 'FormFactor alto';
                    end
                end
            end
        end
        
        % MÉTODO 3: Detección por área grande + geometría razonable
        if ~is_octagon && area > 1000
            if form_factor > 0.70 && solidity > 0.75 && convexity > 0.75
                if aspect_ratio > 0.75 && aspect_ratio < 1.35
                    is_octagon = true;
                    detection_method = 'Área grande + geometría';
                end
            end
        end
        
        if is_octagon
            octagon_mask_red = octagon_mask_red | mask_region;
            num_octagons_red = num_octagons_red + 1;
            fprintf('  ✓ OCTÓGONO ROJO #%d detectado:\n', num_octagons_red);
            fprintf('    - Método: %s\n', detection_method);
            fprintf('    - Área: %d px\n', area);
            fprintf('    - Form Factor: %.3f %s\n', form_factor, tern(form_factor>0.70 && form_factor<0.99));
            fprintf('    - Solidez: %.3f %s\n', solidity, tern(solidity>0.70));
            fprintf('    - Convexidad: %.3f %s\n', convexity, tern(convexity>0.75));
            fprintf('    - Aspect: %.2f %s\n', aspect_ratio, tern(aspect_ratio>0.70 && aspect_ratio<1.45));
            fprintf('    - Extent: %.3f %s\n', extent, tern(extent>0.40));
            fprintf('    - Overlap rojo: %.2f%%\n', overlap_red*100);
            fprintf('    - Vértices detectados: %d\n', best_vertices);
        elseif area > 500 && overlap_red > 0.5
            fprintf('  ✗ Región roja #%d NO clasificada como octógono:\n', idx);
            fprintf('    - Área=%d, FF=%.3f, Sol=%.3f, Conv=%.3f, Asp=%.2f\n', ...
                area, form_factor, solidity, convexity, aspect_ratio);
            fprintf('    - Extent=%.3f, Overlap=%.2f%%\n', extent, overlap_red*100);
            fprintf('    - Vértices máx: %d\n', best_vertices);
        end
    end
    
    fprintf('✓ Total octógonos rojos: %d\n', num_octagons_red);
end

% Función auxiliar
function s = tern(cond)
    if cond
        s = '✓';
    else
        s = '✗';
    end
end

subplot(3,5,10), imshow(octagon_mask_red), title(sprintf('10. Octógonos ROJOS: %d', num_octagons_red));

%% 5. SELECCIÓN DIRECTA DE MEJOR FORMA (SIN FUSIÓN)

fprintf('\n=== SELECCIÓN DE MEJOR DETECCIÓN ===\n');

% Crear lista de detecciones con sus máscaras y scores
detecciones = {};

% 1. OCTÓGONOS ROJOS (máxima prioridad)
if num_octagons_red > 0
    CC = bwconncomp(octagon_mask_red);
    for k = 1:CC.NumObjects
        mask_temp = false(rows, cols);
        mask_temp(CC.PixelIdxList{k}) = true;
        
        stats = regionprops(mask_temp, 'Area', 'Solidity', 'Centroid', 'Extent', 'BoundingBox');
        if ~isempty(stats) && stats(1).Area > 300
            % Score alto para octógonos (prioridad)
            center_img = [cols/2, rows/2];
            dist_center = norm(stats(1).Centroid - center_img);
            normalized_dist = dist_center / norm(center_img);
            
            score = stats(1).Area * stats(1).Solidity * stats(1).Extent * (1.5 - normalized_dist*0.3);
            score = score * 1.5; % Bonus por ser octógono
            
            detecciones{end+1} = struct('mask', mask_temp, 'tipo', 'STOP (Octógono Rojo)', ...
                'score', score, 'area', stats(1).Area, 'stats', stats(1));
            
            fprintf('  Octógono rojo: Área=%d, Score=%.0f\n', stats(1).Area, score);
        end
    end
end

% 2. TRIÁNGULOS AMARILLOS
if num_triangles_yellow > 0
    CC = bwconncomp(triangle_mask_yellow);
    for k = 1:CC.NumObjects
        mask_temp = false(rows, cols);
        mask_temp(CC.PixelIdxList{k}) = true;
        
        stats = regionprops(mask_temp, 'Area', 'Solidity', 'Centroid', 'Extent', 'BoundingBox');
        if ~isempty(stats) && stats(1).Area > 300
            center_img = [cols/2, rows/2];
            dist_center = norm(stats(1).Centroid - center_img);
            normalized_dist = dist_center / norm(center_img);
            
            score = stats(1).Area * stats(1).Solidity * stats(1).Extent * (1.4 - normalized_dist*0.4);
            score = score * 1.2; % Bonus por ser triángulo
            
            detecciones{end+1} = struct('mask', mask_temp, 'tipo', 'Advertencia (Triángulo Amarillo)', ...
                'score', score, 'area', stats(1).Area, 'stats', stats(1));
            
            fprintf('  Triángulo amarillo: Área=%d, Score=%.0f\n', stats(1).Area, score);
        end
    end
end

% 3. CÍRCULOS ROJOS
if num_circles_red > 0
    CC = bwconncomp(circle_mask_red);
    for k = 1:CC.NumObjects
        mask_temp = false(rows, cols);
        mask_temp(CC.PixelIdxList{k}) = true;
        
        stats = regionprops(mask_temp, 'Area', 'Solidity', 'Centroid', 'Extent', 'BoundingBox');
        if ~isempty(stats) && stats(1).Area > 300
            center_img = [cols/2, rows/2];
            dist_center = norm(stats(1).Centroid - center_img);
            normalized_dist = dist_center / norm(center_img);
            
            score = stats(1).Area * stats(1).Solidity * stats(1).Extent * (1.3 - normalized_dist*0.5);
            
            detecciones{end+1} = struct('mask', mask_temp, 'tipo', 'Prohibición (Círculo Rojo)', ...
                'score', score, 'area', stats(1).Area, 'stats', stats(1));
            
            fprintf('  Círculo rojo: Área=%d, Score=%.0f\n', stats(1).Area, score);
        end
    end
end

% 4. CÍRCULOS AZULES
if num_circles_blue > 0
    CC = bwconncomp(circle_mask_blue);
    for k = 1:CC.NumObjects
        mask_temp = false(rows, cols);
        mask_temp(CC.PixelIdxList{k}) = true;
        
        stats = regionprops(mask_temp, 'Area', 'Solidity', 'Centroid', 'Extent', 'BoundingBox');
        if ~isempty(stats) && stats(1).Area > 300
            center_img = [cols/2, rows/2];
            dist_center = norm(stats(1).Centroid - center_img);
            normalized_dist = dist_center / norm(center_img);
            
            score = stats(1).Area * stats(1).Solidity * stats(1).Extent * (1.3 - normalized_dist*0.5);
            
            detecciones{end+1} = struct('mask', mask_temp, 'tipo', 'Obligación (Círculo Azul)', ...
                'score', score, 'area', stats(1).Area, 'stats', stats(1));
            
            fprintf('  Círculo azul: Área=%d, Score=%.0f\n', stats(1).Area, score);
        end
    end
end

%% 6. SELECCIONAR LA MEJOR DETECCIÓN

if isempty(detecciones)
    fprintf('\n⚠️  No se detectaron formas válidas\n');
    mascara_final = false(rows, cols);
    img_final = img;
else
    % Ordenar por score
    scores = zeros(length(detecciones), 1);
    for i = 1:length(detecciones)
        scores(i) = detecciones{i}.score;
    end
    
    [~, idx_best] = max(scores);
    best_detection = detecciones{idx_best};
    
    fprintf('\n✓ Mejor detección seleccionada: %s (Score: %.0f)\n', ...
        best_detection.tipo, best_detection.score);
    
    % Usar directamente la máscara de la forma detectada
    mascara_final = best_detection.mask;
    
    % Aplicar limpieza mínima
    mascara_final = imfill(mascara_final, 'holes');
    mascara_final = imclose(mascara_final, strel('disk', 2));
    
    % Aplicar máscara a la imagen
    img_final = img;
    for c = 1:3
        ch = img_final(:,:,c);
        ch(~mascara_final) = 0;
        img_final(:,:,c) = ch;
    end
    
    % Visualizar con bbox
    bbox = best_detection.stats.BoundingBox;
    img_bbox = insertShape(img, 'Rectangle', bbox, 'Color', 'green', 'LineWidth', 3);
    position = [bbox(1)+5, bbox(2)-15];
    img_bbox = insertText(img_bbox, position, best_detection.tipo, ...
        'FontSize', 14, 'BoxColor', 'green', 'TextColor', 'white');
    
    subplot(3,5,13), imshow(img_bbox), title('13. Detección Final');
    
    fprintf('\n✅ SEÑAL DETECTADA Y CLASIFICADA\n');
    fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
    fprintf('Tipo: %s\n', best_detection.tipo);
    fprintf('Área: %d px (%.1f%%)\n', best_detection.area, best_detection.area/(rows*cols)*100);
    fprintf('Solidez: %.2f\n', best_detection.stats.Solidity);
    fprintf('Score: %.0f\n', best_detection.score);
    fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
    
    % Mostrar todas las detecciones con sus scores
    if length(detecciones) > 1
        fprintf('\n📊 Todas las detecciones:\n');
        for i = 1:length(detecciones)
            marker = '';
            if i == idx_best
                marker = ' ← SELECCIONADA';
            end
            fprintf('  %d. %s - Score: %.0f%s\n', i, detecciones{i}.tipo, ...
                detecciones{i}.score, marker);
        end
    end
end

subplot(3,5,11), imshow(mascara_final), title('11. Máscara Final (Forma Directa)');
subplot(3,5,12), imshow(img_final), title('12. Segmentación Final');

%% 9. RESUMEN VISUAL
figure('Position', [100, 100, 1800, 400]);
subplot(1,6,1), imshow(img), title('Original', 'FontSize', 12);
subplot(1,6,2), imshow(red_mask), title(sprintf('Rojo: C=%d O=%d', num_circles_red, num_octagons_red), 'FontSize', 12);
subplot(1,6,3), imshow(blue_mask), title(sprintf('Azul: C=%d', num_circles_blue), 'FontSize', 12);
subplot(1,6,4), imshow(yellow_mask), title(sprintf('Amarillo: T=%d', num_triangles_yellow), 'FontSize', 12);
subplot(1,6,5), imshow(mascara_final), title('Máscara Final (Forma Directa)', 'FontSize', 12);
subplot(1,6,6), imshow(img_final), title('Segmentación', 'FontSize', 12);

sgtitle('Detección Directa por Forma (Sin Fusión)', 'FontSize', 14, 'FontWeight', 'bold');

%% FUNCIÓN AUXILIAR: Filtrar regiones grandes de fondo
function mask_filtered = filtrar_fondo_grande(mask, rows, cols)
    % Eliminar regiones que ocupan >40% de la imagen (probablemente fondo)
    max_area = rows * cols * 0.60;
    
    CC = bwconncomp(mask);
    stats = regionprops(CC, 'Area');
    
    mask_filtered = mask;
    
    if ~isempty(stats)
        areas = [stats.Area];
        
        for i = 1:length(areas)
            if areas(i) > max_area
                % Eliminar esta región grande
                mask_filtered(CC.PixelIdxList{i}) = false;
            end
        end
    end
    
    % Limpiar restos pequeños
    mask_filtered = bwareaopen(mask_filtered, 200);
end