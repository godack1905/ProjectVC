%% PRÀCTICA VC - RECONEIXEMENT DE SENYALS DE TRÀNSIT
% Funció d'extracció de descriptors amb processament robust

function descriptors = extractDescriptors(img)
    % Extreu 25 descriptors amb processament robust previ
    
    descriptors = zeros(1, 25);
    
    % Verificar que la imagen no esté vacía
    if isempty(img)
        return;
    end
    
    try
        % ------------------------------------
        % FASE 1: PROCESAMIENTO ROBUSTO DE LA IMAGEN
        % ------------------------------------
        
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
        
        %% 2. DETECCIÓN POR COLOR - MEJORADA
        % Detección de rojos
        red_rgb = (R > 0.5) & (G < 0.5) & (B < 0.5);
        red_hsv = (H > 0.94 | H < 0.06) & (S > 0.4) & (V > 0.3);
        red_mask = red_rgb | red_hsv;
        
        % Detección de azules
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
        
        % Filtro de saturación
        saturation_filter = S > 0.3;
        colorMask = colorMask & saturation_filter;
        
        %% 3. DETECCIÓN DE FORMAS
        img_enhanced = imadjust(img_gray);
        edges_all = edge(img_enhanced, 'Canny', [0.08 0.22]);
        
        % Inicializar máscaras
        circle_mask = false(rows, cols);
        triangle_mask = false(rows, cols);
        octagon_mask = false(rows, cols);
        
        %% A) DETECCIÓN DE CÍRCULOS
        radio_min = 15;
        radio_max = round(min(rows, cols) / 2);
        [centers, radii, metric] = imfindcircles(img_enhanced, [radio_min radio_max], ...
            'ObjectPolarity', 'dark', ...
            'Sensitivity', 0.93, ...
            'EdgeThreshold', 0.08);
        
        if ~isempty(centers)
            % FILTRAR: Quedarse solo con el círculo más prometedor
            if length(radii) > 1
                center_img = [cols/2, rows/2];
                circle_scores = zeros(length(radii), 1);
                
                for i = 1:length(radii)
                    dist_to_center = norm([centers(i,1), centers(i,2)] - center_img);
                    max_dist = norm(center_img);
                    centrality = 1 - (dist_to_center / max_dist);
                    
                    [xx, yy] = meshgrid(1:cols, 1:rows);
                    circle_temp = ((xx - centers(i,1)).^2 + (yy - centers(i,2)).^2) <= radii(i)^2;
                    color_overlap = sum(colorMask(:) & circle_temp(:)) / sum(circle_temp(:));
                    
                    circle_scores(i) = metric(i) * 0.3 + ...
                                      (radii(i)/radio_max) * 0.2 + ...
                                      centrality * 0.2 + ...
                                      color_overlap * 0.3;
                end
                
                [~, best_idx] = max(circle_scores);
                centers = centers(best_idx, :);
                radii = radii(best_idx);
            end
            
            % Crear máscara del círculo seleccionado
            for i = 1:length(radii)
                [xx, yy] = meshgrid(1:cols, 1:rows);
                circle_temp = ((xx - centers(i,1)).^2 + (yy - centers(i,2)).^2) <= (radii(i)*1.02)^2;
                circle_mask = circle_mask | circle_temp;
            end
        end
        
        %% B) DETECCIÓN DE TRIÁNGULOS Y OCTÓGONOS
        colorMask_clean = imclose(colorMask, strel('disk', 4));
        colorMask_clean = imfill(colorMask_clean, 'holes');
        colorMask_clean = bwareaopen(colorMask_clean, 200);
        
        CC = bwconncomp(colorMask_clean);
        
        for k = 1:CC.NumObjects
            mask_region = false(rows, cols);
            mask_region(CC.PixelIdxList{k}) = true;
            
            stats = regionprops(mask_region, 'Area', 'Perimeter', 'BoundingBox', ...
                'Solidity', 'Extent', 'ConvexArea', 'Eccentricity', 'MajorAxisLength', ...
                'MinorAxisLength', 'Centroid');
            
            if isempty(stats) || stats(1).Area < 300
                continue;
            end
            
            area = stats(1).Area;
            perimeter = stats(1).Perimeter;
            solidity = stats(1).Solidity;
            convexity = area / stats(1).ConvexArea;
            aspect_ratio = stats(1).BoundingBox(3) / stats(1).BoundingBox(4);
            extent = stats(1).Extent;
            
            % Calcular factor de forma
            if perimeter > 0
                form_factor = (4 * pi * area) / (perimeter^2);
            else
                continue;
            end
            
            % ANÁLISIS DEL CONTORNO
            is_triangle = false;
            is_octagon = false;
            
            boundaries = bwboundaries(mask_region, 'noholes');
            if ~isempty(boundaries)
                boundary = boundaries{1};
                
                % Probar múltiples tolerancias
                for tolerance = [0.008, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
                    simplified = reducepoly(boundary, tolerance);
                    num_vertices = size(simplified, 1);
                    
                    % TRIÁNGULOS: 3-6 vértices
                    if num_vertices >= 3 && num_vertices <= 6
                        ff_ok = (form_factor >= 0.38 && form_factor <= 0.75);
                        sol_ok = (solidity >= 0.70 && solidity <= 0.96);
                        conv_ok = (convexity >= 0.75);
                        asp_ok = (aspect_ratio >= 0.60 && aspect_ratio <= 1.50);
                        ecc_ok = (stats(1).Eccentricity <= 0.70);
                        ext_ok = (extent >= 0.35);
                        
                        score = ff_ok + sol_ok + conv_ok + asp_ok + ecc_ok + ext_ok;
                        
                        if score >= 4
                            is_triangle = true;
                            break;
                        end
                    end
                end
                
                % OCTÓGONOS
                if ~is_triangle && area > 500
                    octagon_ff = (form_factor > 0.77 && form_factor < 0.98);
                    octagon_sol = (solidity > 0.80);
                    
                    if octagon_ff && octagon_sol
                        simplified = reducepoly(boundary, 0.02);
                        num_vertices = size(simplified, 1);
                        
                        if num_vertices >= 6 && num_vertices <= 12
                            is_octagon = true;
                        end
                    end
                end
            end
            
            if is_triangle
                triangle_mask = triangle_mask | mask_region;
            elseif is_octagon
                octagon_mask = octagon_mask | mask_region;
            end
        end
        
        %% 4. FUSIÓN INTELIGENTE
        shape_mask = circle_mask | triangle_mask | octagon_mask;
        
        if sum(shape_mask(:)) > 300
            % Expansión mínima fija
            shape_expanded = imdilate(shape_mask, strel('disk', 1));
            color_dilated = colorMask;
            expansion_zone = shape_expanded & ~shape_mask;
            valid_expansion = expansion_zone & color_dilated;
            fusion_mask = shape_mask | valid_expansion;
            fusion_mask = imfill(fusion_mask, 'holes');
            fusion_mask = imclose(fusion_mask, strel('disk', 1));
        elseif sum(shape_mask(:)) > 100
            shape_expanded = imdilate(shape_mask, strel('disk', 2));
            color_close = imclose(colorMask, strel('disk', 2));
            fusion_mask = shape_expanded & color_close;
            fusion_mask = imfill(fusion_mask, 'holes');
        elseif sum(colorMask(:)) > 400
            fusion_mask = imclose(colorMask, strel('disk', 2));
            fusion_mask = imfill(fusion_mask, 'holes');
            fusion_mask = bwareaopen(fusion_mask, 300);
        else
            fusion_mask = imdilate(colorMask, strel('disk', 2)) | imdilate(shape_mask, strel('disk', 2));
            fusion_mask = imfill(fusion_mask, 'holes');
        end
        
        %% 5. MORFOLOGÍA FINAL
        final_mask = imclose(fusion_mask, strel('disk', 2));
        final_mask = imfill(final_mask, 'holes');
        final_mask = imopen(final_mask, strel('disk', 1));
        final_mask = bwareaopen(final_mask, 200);
        final_mask = imclose(final_mask, strel('disk', 2));
        final_mask = imfill(final_mask, 'holes');
        
        %% 6. SELECCIÓN DE MEJOR REGIÓN
        stats_regions = regionprops(final_mask, 'Area', 'BoundingBox', 'Solidity', ...
            'PixelIdxList', 'Eccentricity', 'Centroid', 'Perimeter', 'Extent');
        
        mascara_final = false(rows, cols);
        
        if ~isempty(stats_regions)
            valid = false(length(stats_regions), 1);
            scores = zeros(length(stats_regions), 1);
            
            for i = 1:length(stats_regions)
                area = stats_regions(i).Area;
                bbox = stats_regions(i).BoundingBox;
                aspect = bbox(3) / bbox(4);
                
                area_ok = area > 300 && area < rows*cols*0.8;
                solidity_ok = stats_regions(i).Solidity > 0.55;
                aspect_ok = aspect > 0.4 && aspect < 2.5;
                extent_ok = stats_regions(i).Extent > 0.3;
                
                valid(i) = area_ok && solidity_ok && aspect_ok && extent_ok;
                
                if valid(i)
                    center_img = [cols/2, rows/2];
                    dist_center = norm(stats_regions(i).Centroid - center_img);
                    normalized_dist = dist_center / norm(center_img);
                    
                    scores(i) = area * stats_regions(i).Solidity * ...
                               stats_regions(i).Extent * (1.3 - normalized_dist*0.5);
                end
            end
            
            stats_regions = stats_regions(valid);
            scores = scores(valid);
            
            if isempty(stats_regions)
                % Tomar la región más grande
                stats_all = regionprops(final_mask, 'Area', 'PixelIdxList');
                if ~isempty(stats_all)
                    [~, idx_max] = max([stats_all.Area]);
                    mascara_final(stats_all(idx_max).PixelIdxList) = true;
                end
            else
                [~, idx] = max(scores);
                mascara_final(stats_regions(idx).PixelIdxList) = true;
            end
            
            % Suavizar máscara final
            mascara_final = imclose(mascara_final, strel('disk', 2));
            mascara_final = imfill(mascara_final, 'holes');
        else
            mascara_final = final_mask;
        end
        
        % Aplicar máscara a la imagen original para segmentación
        img_segmented = img;
        for c = 1:3
            channel = img_segmented(:,:,c);
            channel(~mascara_final) = 0;
            img_segmented(:,:,c) = channel;
        end
        
        % ------------------------------------
        % FASE 2: EXTRACCIÓN DE DESCRIPTORES
        % Sobre la región segmentada
        % ------------------------------------
        
        % Usar la imagen segmentada para extraer descriptores
        img_masked = img_segmented;
        img_gray_masked = rgb2gray(img_masked);
        
        % Obtener componentes de color de la región segmentada
        img_hsv_masked = rgb2hsv(img_masked);
        H_masked = img_hsv_masked(:,:,1);
        S_masked = img_hsv_masked(:,:,2);
        R_masked = img_masked(:,:,1);
        G_masked = img_masked(:,:,2);
        B_masked = img_masked(:,:,3);
        
        % Detección de colores dentro de la máscara
        red1_masked = (R_masked > 0.5) & (G_masked < 0.5) & (B_masked < 0.5);
        red2_masked = (H_masked > 0.95 | H_masked < 0.05) & (S_masked > 0.5) & (img_hsv_masked(:,:,3) > 0.4);
        
        blue1_masked = (B_masked > 0.5) & (R_masked < 0.3) & (G_masked < 0.5);
        blue2_masked = (H_masked > 0.55 & H_masked < 0.7) & (S_masked > 0.4) & (img_hsv_masked(:,:,3) > 0.3);
        
        yellow1_masked = (R_masked > 0.5) & (G_masked > 0.5) & (B_masked < 0.3);
        yellow2_masked = (H_masked > 0.1 & H_masked < 0.2) & (S_masked > 0.4) & (img_hsv_masked(:,:,3) > 0.4);
        orange1_masked = (R_masked > 0.8) & (G_masked > 0.4) & (B_masked < 0.3);
        orange2_masked = (H_masked > 0.05 & H_masked < 0.1) & (S_masked > 0.4) & (img_hsv_masked(:,:,3) > 0.4);
        
        % Usar la máscara final como región de interés
        combi = mascara_final;
        total_pixels = sum(combi(:));
        
        % --------------------------
        % DESCRIPTORES DE COLOR
        % --------------------------
        
        if total_pixels > 0
            % Porcentajes de colores dentro de la región segmentada
            pct_red = sum((red1_masked(combi) | red2_masked(combi))) / total_pixels;
            pct_blue = sum((blue1_masked(combi) | blue2_masked(combi))) / total_pixels;
            pct_yellow = sum((yellow1_masked(combi) | yellow2_masked(combi) | ...
                             orange1_masked(combi) | orange2_masked(combi))) / total_pixels;
            
            % Valores medios de color
            mean_red = mean(R_masked(combi));
            mean_green = mean(G_masked(combi));
            mean_blue = mean(B_masked(combi));
        else
            pct_red = 0; pct_blue = 0; pct_yellow = 0;
            mean_red = 0; mean_green = 0; mean_blue = 0;
        end
        
        % -------------------------
        % DESCRIPTORES DE FORMA
        % -------------------------
        
        if total_pixels > 100
            stats = regionprops(combi, 'Area', 'Perimeter', 'Eccentricity', ...
                                'Solidity', 'Extent', 'BoundingBox', ...
                                'MajorAxisLength', 'MinorAxisLength');
            
            if ~isempty(stats)
                % Tomar la región principal (ya debería ser solo una)
                area = stats(1).Area;
                perimeter = stats(1).Perimeter;
                eccentricity = stats(1).Eccentricity;
                solidity = stats(1).Solidity;
                extent = stats(1).Extent;
                
                if perimeter > 0
                    circularity = (4 * pi * area) / (perimeter^2);
                else
                    circularity = 0;
                end
                
                bbox = stats(1).BoundingBox;
                if bbox(4) > 0
                    aspect_ratio = bbox(3) / bbox(4);
                else
                    aspect_ratio = 0;
                end
                
                if stats(1).MinorAxisLength > 0
                    axis_ratio = stats(1).MajorAxisLength / stats(1).MinorAxisLength;
                else
                    axis_ratio = 0;
                end
                
                if area > 0
                    compactness = perimeter^2 / (4 * pi * area);
                else
                    compactness = 0;
                end
            else
                area = 0; perimeter = 0; eccentricity = 0;
                solidity = 0; extent = 0; circularity = 0;
                aspect_ratio = 0; axis_ratio = 0; compactness = 0;
            end
        else
            area = 0; perimeter = 0; eccentricity = 0;
            solidity = 0; extent = 0; circularity = 0;
            aspect_ratio = 0; axis_ratio = 0; compactness = 0;
        end
        
        % ----------------------------
        % DESCRIPTORES DE FOURIER
        % ----------------------------
        
        fourier_desc = zeros(1, 5);
        if total_pixels > 200
            try
                contorn = bwperim(combi);
                [y, x] = find(contorn);
                
                if length(x) > 10
                    centro = [mean(x), mean(y)];
                    s = (x - centro(1)) + 1i * (y - centro(2));
                    z = fft(s(:));
                    
                    if abs(z(1)) > 0
                        z_norm = z / abs(z(1));
                        fourier_desc = abs(z_norm(1:5))';
                    end
                end
            catch
                fourier_desc = zeros(1, 5);
            end
        end
        
        % ----------------------------
        % DESCRIPTORES DE TEXTURA
        % ----------------------------
        
        if total_pixels > 0
            % Detección de edges en la región segmentada
            edges_masked = edge(img_gray_masked, 'Canny', [0.1 0.2]);
            pct_edges = sum(edges_masked(combi)) / total_pixels;
            
            % GLCM en la región segmentada
            try
                region_gray = img_gray_masked;
                region_gray(~combi) = 0;
                glcm = graycomatrix(region_gray, 'Offset', [0 1], 'Symmetric', true);
                stats_glcm = graycoprops(glcm);
                contrast = stats_glcm.Contrast;
                correlation = stats_glcm.Correlation;
                energy = stats_glcm.Energy;
                homogeneity = stats_glcm.Homogeneity;
            catch
                contrast = 0; correlation = 0; energy = 0; homogeneity = 0;
            end
        else
            pct_edges = 0;
            contrast = 0; correlation = 0; energy = 0; homogeneity = 0;
        end
        
        % ---------------------------------
        % CONSTRUIR VECTOR DE DESCRIPTORES
        % ---------------------------------
        
        descriptors = [
            double(pct_red), double(pct_blue), double(pct_yellow), ...
            double(mean_red), double(mean_green), double(mean_blue), ...
            double(area), double(perimeter), double(eccentricity), ...
            double(solidity), double(extent), double(circularity), ...
            double(aspect_ratio), double(axis_ratio), double(compactness), ...
            double(fourier_desc(1)), double(fourier_desc(2)), ...
            double(fourier_desc(3)), double(fourier_desc(4)), ...
            double(fourier_desc(5)), ...
            double(pct_edges), double(contrast), double(correlation), ...
            double(energy), double(homogeneity)
        ];
        
    catch ME
        fprintf('Error en extractDescriptors: %s\n', ME.message);
        descriptors = zeros(1, 25);
    end
end