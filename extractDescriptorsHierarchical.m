function [desc_forma, desc_color, desc_detall, forma_type, color_type] = extractDescriptorsHierarchical(img)
    % Extrae descriptors para los 5 modelos jerárquicos
    
    % Inicializar descriptores
    desc_forma = zeros(1, 16);
    desc_color = zeros(1, 15);
    desc_detall = zeros(1, 30);
    forma_type = 'unknown';
    color_type = 'unknown';
    
    try
        %% SEGMENTACIÓN BÁSICA
        [rows, cols, ~] = size(img);
        img_hsv = rgb2hsv(img);
        img_gray = rgb2gray(img);
        
        % Detección de colores básicos
        H = img_hsv(:,:,1);
        S = img_hsv(:,:,2);
        V = img_hsv(:,:,3);
        
        % Normalizar RGB
        VAL = double(ones(rows, cols));
        normalized_hsv = cat(3, H, S, VAL);
        rgb_norm = hsv2rgb(normalized_hsv);
        R = rgb_norm(:,:,1);
        G = rgb_norm(:,:,2);
        B = rgb_norm(:,:,3);
        
        % Máscaras de color
        red_mask = (R > 0.5) & (G < 0.5) & (B < 0.5) | ...
                   (H > 0.94 | H < 0.06) & (S > 0.4) & (V > 0.3);
        
        blue_mask = (B > 0.5) & (R < 0.4) & (G < 0.5) | ...
                    (H > 0.54 & H < 0.70) & (S > 0.35) & (V > 0.25);
        
        yellow_mask = (R > 0.5) & (G > 0.45) & (B < 0.35) | ...
                      (H > 0.10 & H < 0.20) & (S > 0.4) & (V > 0.35);
        
        white_mask = (R > 0.7) & (G > 0.7) & (B > 0.7) & (S < 0.3);
        
        % Máscara combinada
        color_mask = red_mask | blue_mask | yellow_mask | white_mask;
        color_mask = color_mask & (S > 0.2);
        color_mask = imclose(color_mask, strel('disk', 3));
        color_mask = imfill(color_mask, 'holes');
        color_mask = bwareaopen(color_mask, 200);
        
        %% EXTRACCIÓN DE REGIÓN DE INTERÉS
        stats = regionprops(color_mask, 'Area', 'BoundingBox', 'PixelIdxList');
        if isempty(stats)
            return;
        end
        
        % Seleccionar región más grande
        [~, idx] = max([stats.Area]);
        roi_mask = false(rows, cols);
        roi_mask(stats(idx).PixelIdxList) = true;
        
        % Crear imagen segmentada
        img_seg = img;
        for c = 1:3
            channel = img_seg(:,:,c);
            channel(~roi_mask) = 0;
            img_seg(:,:,c) = channel;
        end
        
        %% DESCRIPTORES DE FORMA (Modelo 1)
        stats_forma = regionprops(roi_mask, 'Area', 'Perimeter', 'Eccentricity', ...
            'Solidity', 'Extent', 'BoundingBox', 'MajorAxisLength', ...
            'MinorAxisLength', 'ConvexArea', 'EquivDiameter');
        
        if ~isempty(stats_forma)
            area = stats_forma(1).Area;
            perimeter = stats_forma(1).Perimeter;
            
            % Factor de forma (circularidad)
            if perimeter > 0
                circularity = (4 * pi * area) / (perimeter^2);
            else
                circularity = 0;
            end
            
            % Rectangularidad
            bbox = stats_forma(1).BoundingBox;
            bbox_area = bbox(3) * bbox(4);
            if bbox_area > 0
                rectangularity = area / bbox_area;
            else
                rectangularity = 0;
            end
            
            % Análisis de contorno para número de vértices
            boundary = bwboundaries(roi_mask, 'noholes');
            if ~isempty(boundary)
                simplified = reducepoly(boundary{1}, 0.02);
                num_vertices = size(simplified, 1);
            else
                num_vertices = 0;
            end
            
            % Descriptores de Fourier para forma
            fourier_desc = zeros(1, 3);
            try
                contorn = bwperim(roi_mask);
                [y, x] = find(contorn);
                if length(x) > 10
                    centro = [mean(x), mean(y)];
                    s = (x - centro(1)) + 1i * (y - centro(2));
                    z = fft(s(:));
                    if abs(z(1)) > 0
                        z_norm = z / abs(z(1));
                        fourier_desc = abs(z_norm(1:3))';
                    end
                end
            catch
                fourier_desc = zeros(1, 3);
            end
            
            % Determinar tipo de forma
            if circularity > 0.85
                forma_type = 'circular';
            elseif circularity > 0.4 && circularity < 0.7
                forma_type = 'triangular';
            else
                forma_type = 'other';
            end
            
            % Vector de descriptores de forma
            desc_forma = [
                circularity, stats_forma(1).Eccentricity, ...
                stats_forma(1).Solidity, stats_forma(1).Extent, ...
                bbox(3)/bbox(4), ...
                stats_forma(1).MajorAxisLength / max(stats_forma(1).MinorAxisLength, 1), ...
                perimeter^2 / (4 * pi * max(area, 1)), ...
                circularity, num_vertices, fourier_desc, ...
                perimeter, area, rectangularity, stats_forma(1).EquivDiameter
            ];
        end
        
        %% DESCRIPTORES DE COLOR (Modelo 2)
        % Estadísticas de color dentro de la ROI
        R_roi = R(roi_mask);
        G_roi = G(roi_mask);
        B_roi = B(roi_mask);
        H_roi = H(roi_mask);
        S_roi = S(roi_mask);
        V_roi = V(roi_mask);
        
        if ~isempty(R_roi)
            % Porcentajes de colores
            pct_red = sum(red_mask(roi_mask)) / sum(roi_mask(:));
            pct_blue = sum(blue_mask(roi_mask)) / sum(roi_mask(:));
            pct_white = sum(white_mask(roi_mask)) / sum(roi_mask(:));
            pct_yellow = sum(yellow_mask(roi_mask)) / sum(roi_mask(:));
            
            % Estadísticas de color
            mean_red = mean(R_roi);
            mean_green = mean(G_roi);
            mean_blue = mean(B_roi);
            mean_hue = mean(H_roi);
            mean_saturation = mean(S_roi);
            mean_value = mean(V_roi);
            
            std_red = std(double(R_roi));
            std_blue = std(double(B_roi));
            
            % Ratios de color
            red_blue_ratio = mean_red / max(mean_blue, 0.01);
            red_green_ratio = mean_red / max(mean_green, 0.01);
            
            % Entropía de color
            color_hist = histcounts(H_roi, 0:0.1:1);
            color_hist = color_hist / sum(color_hist);
            color_entropy = -sum(color_hist .* log2(color_hist + eps));
            
            % Determinar color dominante
            color_scores = [pct_red, pct_blue, pct_white];
            [~, color_idx] = max(color_scores);
            color_names = {'red', 'blue', 'white'};
            color_type = color_names{color_idx};
            
            % Vector de descriptores de color
            desc_color = [
                pct_red, pct_blue, pct_white, pct_yellow, ...
                mean_red, mean_green, mean_blue, mean_hue, ...
                mean_saturation, mean_value, std_red, std_blue, ...
                red_blue_ratio, red_green_ratio, color_entropy
            ];
        end
        
        %% DESCRIPTORES DETALLADOS (Modelos 3-5)
        % Descriptores más detallados para clasificación específica
        if ~isempty(R_roi)
            % Porcentajes adicionales
            pct_black = sum(V_roi < 0.2) / length(V_roi);
            
            % Estadísticas adicionales
            std_green = std(double(G_roi));
            hue_std = std(H_roi);
            sat_std = std(S_roi);
            
            % Simetría
            try
                symmetry_x = computeSymmetry(roi_mask, 'horizontal');
                symmetry_y = computeSymmetry(roi_mask, 'vertical');
            catch
                symmetry_x = 0;
                symmetry_y = 0;
            end
            
            % Texturas (simplificado)
            img_gray_roi = img_gray;
            img_gray_roi(~roi_mask) = 0;
            
            try
                glcm = graycomatrix(img_gray_roi, 'Offset', [0 1; -1 1; -1 0; -1 -1], 'Symmetric', true);
                stats_glcm = graycoprops(glcm);
                
                contrast = mean([stats_glcm.Contrast]);
                correlation = mean([stats_glcm.Correlation]);
                energy = mean([stats_glcm.Energy]);
                homogeneity = mean([stats_glcm.Homogeneity]);
                
                % Textons simples (promedio de patches)
                patch1 = mean(mean(img_gray_roi(1:end/2, 1:end/2)));
                patch2 = mean(mean(img_gray_roi(1:end/2, end/2:end)));
                patch1 = patch1 / 255;
                patch2 = patch2 / 255;
            catch
                contrast = 0; correlation = 0; energy = 0; homogeneity = 0;
                patch1 = 0; patch2 = 0;
            end
            
            % Descriptores de Fourier adicionales
            fourier_detall = zeros(1, 4);
            try
                contorn = bwperim(roi_mask);
                [y, x] = find(contorn);
                if length(x) > 10
                    centro = [mean(x), mean(y)];
                    s = (x - centro(1)) + 1i * (y - centro(2));
                    z = fft(s(:));
                    if abs(z(1)) > 0
                        z_norm = z / abs(z(1));
                        fourier_detall = abs(z_norm(1:4))';
                    end
                end
            catch
                fourier_detall = zeros(1, 4);
            end
            
            % Porcentaje de edges
            edges_roi = edge(img_gray_roi, 'Canny');
            pct_edges = sum(edges_roi(roi_mask)) / sum(roi_mask(:));
            
            % Vector de descriptores detallados
            desc_detall = [
                pct_red, pct_blue, pct_white, pct_yellow, pct_black, ...
                mean_red, mean_green, mean_blue, std_red, std_green, ...
                mean_hue, hue_std, mean_saturation, sat_std, ...
                circularity, stats_forma(1).Solidity, stats_forma(1).Extent, ...
                perimeter^2 / (4 * pi * max(area, 1)), ...
                fourier_detall, pct_edges, contrast, energy, homogeneity, ...
                symmetry_x, symmetry_y, patch1, patch2
            ];
        end
        
    catch ME
        fprintf('Error en extractDescriptorsHierarchical: %s\n', ME.message);
    end
end

function symmetry = computeSymmetry(mask, direction)
    % Calcula simetría de una máscara
    [rows, cols] = size(mask);
    
    if strcmp(direction, 'horizontal')
        half = floor(cols/2);
        left_half = mask(:, 1:half);
        right_half = fliplr(mask(:, cols-half+1:cols));
        overlap = left_half & right_half;
        symmetry = sum(overlap(:)) / min(sum(left_half(:)), sum(right_half(:)));
    else % vertical
        half = floor(rows/2);
        top_half = mask(1:half, :);
        bottom_half = flipud(mask(rows-half+1:rows, :));
        overlap = top_half & bottom_half;
        symmetry = sum(overlap(:)) / min(sum(top_half(:)), sum(bottom_half(:)));
    end
end