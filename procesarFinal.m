function mascara_final = procesarFinal(img)
    
    [rows, cols, ~] = size(img);
    
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
    
    %% 2. DETECCIÓN POR COLOR - MÁSCARAS SEPARADAS
    
    % ===== MÁSCARA ROJA =====
    red_rgb = (R > 0.5) & (G < 0.5) & (B < 0.5);
    red_hsv = (H > 0.94 | H < 0.06) & (S > 0.4) & (V > 0.3);
    red_mask = (red_rgb | red_hsv) & (S > 0.3);
    
    % ===== MÁSCARA AZUL =====
    blue_rgb = (B > 0.5) & (R < 0.4) & (G < 0.5);
    blue_hsv = (H > 0.54 & H < 0.70) & (S > 0.35) & (V > 0.25);
    blue_mask_raw = (blue_rgb | blue_hsv) & (S > 0.3);
    blue_mask = filtrar_fondo_grande(blue_mask_raw, rows, cols);
    
    % ===== MÁSCARA AMARILLA =====
    yellow_rgb = (R > 0.5) & (G > 0.45) & (B < 0.35);
    yellow_hsv = (H > 0.10 & H < 0.20) & (S > 0.4) & (V > 0.35);
    orange_rgb = (R > 0.7) & (G > 0.3 & G < 0.7) & (B < 0.35);
    orange_hsv = (H > 0.04 & H < 0.11) & (S > 0.4) & (V > 0.35);
    yellow_mask_raw = (yellow_rgb | yellow_hsv | orange_rgb | orange_hsv) & (S > 0.3);
    yellow_mask = filtrar_fondo_grande(yellow_mask_raw, rows, cols);
    
    %% 3. DETECCIÓN DE BORDES
    img_enhanced = imadjust(img_gray);
    edges_all = edge(img_enhanced, 'Canny', [0.08 0.22]);
    
    %% 4. DETECCIÓN DE FORMAS POR COLOR
    
    num_circles_red = 0;
    num_circles_blue = 0;
    num_triangles_yellow = 0;
    num_octagons_red = 0;
    
    circle_mask_red = false(rows, cols);
    circle_mask_blue = false(rows, cols);
    triangle_mask_yellow = false(rows, cols);
    octagon_mask_red = false(rows, cols);
    
    % ═══════════════════════════════════════════════════════════
    % A) CÍRCULOS ROJOS
    % ═══════════════════════════════════════════════════════════
    radio_min = 15;
    radio_max = round(min(rows, cols) / 2);
    
    if sum(red_mask(:)) > 300
        % Método mejorado (perímetro)
        red_mask_clean = imclose(red_mask, strel('disk', 5));
        red_mask_clean = imfill(red_mask_clean, 'holes');
        red_mask_clean = bwareaopen(red_mask_clean, 300);
        CC = bwconncomp(red_mask_clean);
        
        if CC.NumObjects > 0
            numPixels = cellfun(@numel, CC.PixelIdxList);
            [~, idx_largest] = max(numPixels);
            red_mask_largest = false(rows, cols);
            red_mask_largest(CC.PixelIdxList{idx_largest}) = true;
            red_boundary = bwperim(red_mask_largest);
            red_boundary = imdilate(red_boundary, strel('disk', 3));
            img_red_edge = img_enhanced;
            img_red_edge(~red_boundary) = 255;
            [centers, radii, ~] = imfindcircles(img_red_edge, [radio_min radio_max], ...
                'ObjectPolarity', 'dark', ...
                'Sensitivity', 0.97, ...
                'EdgeThreshold', 0.02, ...
                'Method', 'TwoStage');
            
            if ~isempty(centers)
                [~, best_idx] = max(radii);
                centers = centers(best_idx, :);
                radii = radii(best_idx);
                num_circles_red = 1;
                [xx, yy] = meshgrid(1:cols, 1:rows);
                circle_temp = ((xx - centers(1)).^2 + (yy - centers(2)).^2) <= (radii*1.02)^2;
                circle_mask_red = circle_temp;
            end
        end
    end
    
    % ═══════════════════════════════════════════════════════════
    % B) CÍRCULOS AZULES
    % ═══════════════════════════════════════════════════════════
    
    if sum(blue_mask(:)) > 300
        blue_mask_clean = imclose(blue_mask, strel('disk', 5));
        blue_mask_clean = imfill(blue_mask_clean, 'holes');
        blue_mask_clean = bwareaopen(blue_mask_clean, 300);
        
        CC = bwconncomp(blue_mask_clean);
        numPixels = cellfun(@numel, CC.PixelIdxList);
        [~, idx_largest] = max(numPixels);
        
        blue_mask_largest = false(rows, cols);
        blue_mask_largest(CC.PixelIdxList{idx_largest}) = true;
        
        blue_boundary = bwperim(blue_mask_largest);
        blue_boundary = imdilate(blue_boundary, strel('disk', 3));
        
        img_blue_edge = img_enhanced;
        img_blue_edge(~blue_boundary) = 255;
        
        radio_min_blue = 20;
        
        [centers_blue, radii_blue, ~] = imfindcircles(img_blue_edge, [radio_min_blue radio_max], ...
            'ObjectPolarity', 'dark', ...
            'Sensitivity', 0.97, ...
            'EdgeThreshold', 0.02, ...
            'Method', 'TwoStage');
        
        if ~isempty(centers_blue)
            [~, idx_best] = max(radii_blue);
            centers_blue = centers_blue(idx_best, :);
            radii_blue = radii_blue(idx_best);
            
            num_circles_blue = 1;
            [xx, yy] = meshgrid(1:cols, 1:rows);
            circle_temp = ((xx - centers_blue(1)).^2 + (yy - centers_blue(2)).^2) <= (radii_blue*1.02)^2;
            circle_mask_blue = circle_temp;
        end
    end
    
    % ═══════════════════════════════════════════════════════════
    % C) BÚSQUEDA DE RESPALDO: Círculos en imagen original
    %    (Solo si no se encontraron círculos rojos ni azules)
    % ═══════════════════════════════════════════════════════════
    if num_circles_red == 0 && num_circles_blue == 0
        [centers, radii, metric] = imfindcircles(img_enhanced, [radio_min radio_max], ...
            'ObjectPolarity', 'dark', 'Sensitivity', 0.93, 'EdgeThreshold', 0.08);
        
        if ~isempty(centers)
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
                end
                [~, best_idx] = max(circle_scores);
                centers = centers(best_idx, :);
                radii = radii(best_idx);
            end
            num_circles_red = length(radii);
            for i = 1:num_circles_red
                [xx, yy] = meshgrid(1:cols, 1:rows);
                circle_temp = ((xx - centers(i,1)).^2 + (yy - centers(i,2)).^2) <= (radii(i)*1.02)^2;
                circle_mask_red = circle_mask_red | circle_temp;
            end
        end
    end
    
    % ═══════════════════════════════════════════════════════════
    % D) TRIÁNGULOS AMARILLOS
    % ═══════════════════════════════════════════════════════════
    
    if sum(yellow_mask(:)) > 300
        yellow_mask_clean = imclose(yellow_mask, strel('disk', 4));
        yellow_mask_clean = imfill(yellow_mask_clean, 'holes');
        
        yellow_mask_dilated = imdilate(yellow_mask_clean, strel('disk', 3));
        yellow_mask_dilated = imfill(yellow_mask_dilated, 'holes');
        
        yellow_mask_final = imerode(yellow_mask_dilated, strel('disk', 3));
        yellow_mask_final = bwareaopen(yellow_mask_final, 300);
        
        yellow_mask_edges = imdilate(yellow_mask, strel('disk', 3));
        edges_yellow = edges_all & yellow_mask_edges;
        
        edges_yellow_clean = imclose(edges_yellow, strel('disk', 3));
        edges_yellow_clean = imfill(edges_yellow_clean, 'holes');
        edges_yellow_clean = bwareaopen(edges_yellow_clean, 300);
        
        CC_mask = bwconncomp(yellow_mask_final);
        CC_edges = bwconncomp(edges_yellow_clean);
        
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
            
            overlap_yellow = sum(yellow_mask(:) & mask_region(:)) / area;
            
            if overlap_yellow < 0.3
                continue;
            end
            
            is_triangle = false;
            
            % MÉTODO 1: Análisis del contorno EXTERNO
            boundaries = bwboundaries(mask_region);
            
            if ~isempty(boundaries)
                boundary = boundaries{1};
                
                for tolerance = [0.005, 0.008, 0.01, 0.012, 0.015, 0.018, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045]
                    simplified = reducepoly(boundary, tolerance);
                    num_vertices = size(simplified, 1);
                    
                    if num_vertices >= 3 && num_vertices <= 7
                        ff_ok = (form_factor >= 0.30 && form_factor <= 0.78);
                        sol_ok = (solidity >= 0.65);
                        conv_ok = (convexity >= 0.65);
                        asp_ok = (aspect_ratio >= 0.55 && aspect_ratio <= 1.60);
                        ext_ok = (extent >= 0.35);
                        
                        score = ff_ok + sol_ok + conv_ok + asp_ok + ext_ok;
                        
                        if score >= 3
                            is_triangle = true;
                            break;
                        end
                    end
                end
            end
            
            % MÉTODO 2: Análisis del casco convexo
            if ~is_triangle
                [B_y, B_x] = find(mask_region);
                if length(B_x) > 10
                    try
                        k = convhull(B_x, B_y);
                        convex_boundary = [B_x(k), B_y(k)];
                        
                        for tolerance = [0.01, 0.015, 0.02, 0.025, 0.03]
                            simplified_convex = reducepoly(convex_boundary, tolerance);
                            num_vertices_convex = size(simplified_convex, 1);
                            
                            if num_vertices_convex >= 3 && num_vertices_convex <= 6
                                if form_factor >= 0.30 && solidity >= 0.60 && convexity >= 0.60
                                    is_triangle = true;
                                    break;
                                end
                            end
                        end
                    catch
                        % Ignorar errores
                    end
                end
            end
            
            % MÉTODO 3: Detección por esquinas
            if ~is_triangle && area > 400
                mask_uint8 = uint8(mask_region) * 255;
                corners = detectHarrisFeatures(mask_uint8, 'MinQuality', 0.01);
                
                if corners.Count >= 3 && corners.Count <= 9
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
                    
                    if corners_on_boundary >= 3 && corners_on_boundary <= 6
                        if form_factor >= 0.28 && solidity >= 0.60 && convexity >= 0.60
                            is_triangle = true;
                        end
                    end
                end
            end
            
            % MÉTODO 4: Criterios geométricos puros
            if ~is_triangle
                eccentricity = stats(1).Eccentricity;
                
                is_triangular_shape = (form_factor >= 0.32 && form_factor <= 0.72) && ...
                                      (solidity >= 0.65 && solidity <= 0.94) && ...
                                      (convexity >= 0.65) && ...
                                      (eccentricity >= 0.3 && eccentricity <= 0.95) && ...
                                      (aspect_ratio >= 0.60 && aspect_ratio <= 1.50) && ...
                                      (extent >= 0.38);
                
                if is_triangular_shape
                    is_triangle = true;
                end
            end
            
            if is_triangle
                triangle_mask_yellow = triangle_mask_yellow | mask_region;
                num_triangles_yellow = num_triangles_yellow + 1;
            end
        end
    end
    
    % ═══════════════════════════════════════════════════════════
    % E) OCTÓGONOS ROJOS
    % ═══════════════════════════════════════════════════════════
    
    if sum(red_mask(:)) > 500
        red_mask_clean = imclose(red_mask, strel('disk', 2));
        red_mask_clean = imfill(red_mask_clean, 'holes');
        red_mask_clean = bwareaopen(red_mask_clean, 300);
        
        red_mask_dilated = imdilate(red_mask, strel('disk', 5));
        edges_red = edges_all & red_mask_dilated;
        edges_red_clean = imclose(edges_red, strel('disk', 2));
        edges_red_clean = imfill(edges_red_clean, 'holes');
        edges_red_clean = bwareaopen(edges_red_clean, 300);
        
        CC_red_mask = bwconncomp(red_mask_clean);
        CC_red_edges = bwconncomp(edges_red_clean);
        
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
            
            overlap_red = sum(red_mask(:) & mask_region(:)) / area;
            
            if overlap_red < 0.4
                continue;
            end
            
            is_octagon = false;
            
            % MÉTODO 1: Análisis de vértices
            boundaries = bwboundaries(mask_region, 'noholes');
            
            if ~isempty(boundaries)
                boundary = boundaries{1};
                
                for tolerance = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
                    simplified = reducepoly(boundary, tolerance);
                    num_vertices = size(simplified, 1);
                    
                    if num_vertices >= 6 && num_vertices <= 14
                        ff_ok = (form_factor > 0.70 && form_factor < 0.99);
                        sol_ok = (solidity > 0.70);
                        conv_ok = (convexity > 0.75);
                        asp_ok = (aspect_ratio > 0.70 && aspect_ratio < 1.45);
                        ext_ok = (extent > 0.40);
                        
                        score = ff_ok + sol_ok + conv_ok + asp_ok + ext_ok;
                        
                        if score >= 3
                            is_octagon = true;
                            break;
                        end
                    end
                end
            end
            
            % MÉTODO 2: Form Factor alto
            if ~is_octagon
                if form_factor > 0.75 && form_factor < 0.96
                    if solidity > 0.80 && convexity > 0.80 && extent > 0.45
                        if num_circles_red == 0 || area > sum(circle_mask_red(:)) * 0.5
                            is_octagon = true;
                        end
                    end
                end
            end
            
            % MÉTODO 3: Área grande + geometría
            if ~is_octagon && area > 1000
                if form_factor > 0.70 && solidity > 0.75 && convexity > 0.75
                    if aspect_ratio > 0.75 && aspect_ratio < 1.35
                        is_octagon = true;
                    end
                end
            end
            
            if is_octagon
                octagon_mask_red = octagon_mask_red | mask_region;
                num_octagons_red = num_octagons_red + 1;
            end
        end
    end
    
    %% 5. SELECCIÓN DIRECTA DE MEJOR FORMA
    
    detecciones = {};
    
    % 1. OCTÓGONOS ROJOS (máxima prioridad)
    if num_octagons_red > 0
        CC = bwconncomp(octagon_mask_red);
        for k = 1:CC.NumObjects
            mask_temp = false(rows, cols);
            mask_temp(CC.PixelIdxList{k}) = true;
            
            stats = regionprops(mask_temp, 'Area', 'Solidity', 'Centroid', 'Extent', 'BoundingBox');
            if ~isempty(stats) && stats(1).Area > 300
                center_img = [cols/2, rows/2];
                dist_center = norm(stats(1).Centroid - center_img);
                normalized_dist = dist_center / norm(center_img);
                
                score = stats(1).Area * stats(1).Solidity * stats(1).Extent * (1.5 - normalized_dist*0.3);
                score = score * 1.5;
                
                detecciones{end+1} = struct('mask', mask_temp, 'tipo', 'STOP (Octógono Rojo)', ...
                    'score', score, 'area', stats(1).Area, 'stats', stats(1));
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
                score = score * 1.2;
                
                detecciones{end+1} = struct('mask', mask_temp, 'tipo', 'Advertencia (Triángulo Amarillo)', ...
                    'score', score, 'area', stats(1).Area, 'stats', stats(1));
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
            end
        end
    end
    
    %% 6. SELECCIONAR LA MEJOR DETECCIÓN
    
    if isempty(detecciones)
        mascara_final = false(rows, cols);
    else
        scores = zeros(length(detecciones), 1);
        for i = 1:length(detecciones)
            scores(i) = detecciones{i}.score;
        end
        
        [~, idx_best] = max(scores);
        best_detection = detecciones{idx_best};
        
        mascara_final = best_detection.mask;
        
        mascara_final = imfill(mascara_final, 'holes');
        mascara_final = imclose(mascara_final, strel('disk', 2));
    end
end

%% FUNCIÓN AUXILIAR
function mask_filtered = filtrar_fondo_grande(mask, rows, cols)
    max_area = rows * cols * 0.60;
    
    CC = bwconncomp(mask);
    stats = regionprops(CC, 'Area');
    
    mask_filtered = mask;
    
    if ~isempty(stats)
        areas = [stats.Area];
        
        for i = 1:length(areas)
            if areas(i) > max_area
                mask_filtered(CC.PixelIdxList{i}) = false;
            end
        end
    end
    
    mask_filtered = bwareaopen(mask_filtered, 200);
end