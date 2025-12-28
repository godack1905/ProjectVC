function [desc_forma, desc_color, desc_detall] = extractDescriptors5Models(img)
    % Extreu descriptors per als 5 models jeràrquics
    
    % Inicialitzar
    desc_forma = zeros(1, 16);
    desc_color = zeros(1, 15);
    desc_detall = zeros(1, 30);
    
    try
        %% 1. PREPROCESSAMENT BÀSIC
        [rows, cols, ~] = size(img);
        img_hsv = rgb2hsv(img);
        img_gray = rgb2gray(img);
        
        H = img_hsv(:,:,1);
        S = img_hsv(:,:,2);
        V = img_hsv(:,:,3);
        
        % Normalitzar RGB
        VAL = ones(rows, cols, 'double');
        normalized_hsv = cat(3, H, S, VAL);
        rgb_norm = hsv2rgb(normalized_hsv);
        R = rgb_norm(:,:,1);
        G = rgb_norm(:,:,2);
        B = rgb_norm(:,:,3);
        
        %% 2. SEGMENTACIÓ PER COLOR
        % Màscares de color
        red_mask = (R > 0.5) & (G < 0.5) & (B < 0.5);
        blue_mask = (B > 0.5) & (R < 0.4) & (G < 0.5);
        yellow_mask = (R > 0.5) & (G > 0.45) & (B < 0.35);
        white_mask = (R > 0.7) & (G > 0.7) & (B > 0.7) & (S < 0.3);
        
        color_mask = red_mask | blue_mask | yellow_mask | white_mask;
        color_mask = color_mask & (S > 0.2);
        color_mask = imclose(color_mask, strel('disk', 2));
        color_mask = imfill(color_mask, 'holes');
        color_mask = bwareaopen(color_mask, 150);
        
        %% 3. EXTRACCIÓ REGIÓ D'INTERÈS
        stats = regionprops(color_mask, 'Area', 'PixelIdxList');
        if isempty(stats)
            return;
        end
        
        [~, idx] = max([stats.Area]);
        roi_mask = false(rows, cols);
        roi_mask(stats(idx).PixelIdxList) = true;
        
        % Imatge segmentada
        img_seg = img;
        for c = 1:3
            ch = img_seg(:,:,c);
            ch(~roi_mask) = 0;
            img_seg(:,:,c) = ch;
        end
        
        %% 4. DESCRIPTORS DE FORMA (Model 1)
        stats_forma = regionprops(roi_mask, 'Area', 'Perimeter', 'Eccentricity', ...
            'Solidity', 'Extent', 'BoundingBox', 'MajorAxisLength', ...
            'MinorAxisLength', 'ConvexArea', 'EquivDiameter');
        
        if ~isempty(stats_forma)
            area = stats_forma(1).Area;
            perimeter = stats_forma(1).Perimeter;
            bbox = stats_forma(1).BoundingBox;
            
            % Circularitat
            circularity = (4 * pi * area) / (perimeter^2);
            
            % Rectangularitat
            bbox_area = bbox(3) * bbox(4);
            rectangularity = area / bbox_area;
            
            % Nombre de vèrtexs
            boundary = bwboundaries(roi_mask, 'noholes');
            num_vertices = 0;
            if ~isempty(boundary)
                simplified = reducepoly(boundary{1}, 0.02);
                num_vertices = size(simplified, 1);
            end
            
            % Fourier per forma
            fourier_forma = zeros(1, 3);
            try
                contorn = bwperim(roi_mask);
                [y, x] = find(contorn);
                if length(x) > 10
                    centro = [mean(x), mean(y)];
                    s = (x - centro(1)) + 1i * (y - centro(2));
                    z = fft(s(:));
                    if abs(z(1)) > 0
                        z_norm = z / abs(z(1));
                        fourier_forma = abs(z_norm(1:3))';
                    end
                end
            catch
            end
            
            % Descriptors de forma
            desc_forma = [
                circularity, stats_forma(1).Eccentricity, ...
                stats_forma(1).Solidity, stats_forma(1).Extent, ...
                bbox(3)/bbox(4), ...
                stats_forma(1).MajorAxisLength / max(stats_forma(1).MinorAxisLength, 1), ...
                perimeter^2 / (4 * pi * area), circularity, ...
                num_vertices, fourier_forma, perimeter, area, ...
                rectangularity, stats_forma(1).EquivDiameter
            ];
        end
        
        %% 5. DESCRIPTORS DE COLOR (Model 2)
        % Estadístiques dins la ROI
        R_roi = R(roi_mask);
        G_roi = G(roi_mask);
        B_roi = B(roi_mask);
        H_roi = H(roi_mask);
        S_roi = S(roi_mask);
        V_roi = V(roi_mask);
        
        if ~isempty(R_roi)
            % Percentatges
            pct_red = sum(red_mask(roi_mask)) / sum(roi_mask(:));
            pct_blue = sum(blue_mask(roi_mask)) / sum(roi_mask(:));
            pct_white = sum(white_mask(roi_mask)) / sum(roi_mask(:));
            pct_yellow = sum(yellow_mask(roi_mask)) / sum(roi_mask(:));
            
            % Mitjanes
            mean_red = mean(R_roi);
            mean_green = mean(G_roi);
            mean_blue = mean(B_roi);
            mean_hue = mean(H_roi);
            mean_saturation = mean(S_roi);
            mean_value = mean(V_roi);
            
            % Desviacions
            std_red = std(double(R_roi));
            std_blue = std(double(B_roi));
            
            % Ratios
            red_blue_ratio = mean_red / max(mean_blue, 0.01);
            red_green_ratio = mean_red / max(mean_green, 0.01);
            
            % Entropia de color
            color_hist = histcounts(H_roi, 0:0.1:1);
            color_hist = color_hist / sum(color_hist);
            color_entropy = -sum(color_hist .* log2(color_hist + eps));
            
            % Descriptors de color
            desc_color = [
                pct_red, pct_blue, pct_white, pct_yellow, ...
                mean_red, mean_green, mean_blue, mean_hue, ...
                mean_saturation, mean_value, std_red, std_blue, ...
                red_blue_ratio, red_green_ratio, color_entropy
            ];
        end
        
        %% 6. DESCRIPTORS DETALLATS (Models 3-5)
        if ~isempty(R_roi)
            % Percentatges addicionals
            pct_black = sum(V_roi < 0.2) / length(V_roi);
            
            % Estadístiques addicionals
            std_green = std(double(G_roi));
            hue_std = std(H_roi);
            sat_std = std(S_roi);
            
            % Simetria
            symmetry_x = computeSymmetrySimple(roi_mask, 'horizontal');
            symmetry_y = computeSymmetrySimple(roi_mask, 'vertical');
            
            % Textura
            img_gray_roi = img_gray;
            img_gray_roi(~roi_mask) = 0;
            
            contrast = 0; energy = 0; homogeneity = 0;
            try
                glcm = graycomatrix(img_gray_roi, 'Offset', [0 1], 'Symmetric', true);
                stats_glcm = graycoprops(glcm);
                contrast = stats_glcm.Contrast;
                energy = stats_glcm.Energy;
                homogeneity = stats_glcm.Homogeneity;
            catch
            end
            
            % Fourier addicional
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
            end
            
            % Edges
            edges_roi = edge(img_gray_roi, 'Canny', [0.1 0.2]);
            pct_edges = sum(edges_roi(roi_mask)) / sum(roi_mask(:));
            
            % Textons simples
            img_gray_roi_double = double(img_gray_roi) / 255;
            texton1 = mean(img_gray_roi_double(1:floor(end/2), :), 'all');
            texton2 = mean(img_gray_roi_double(floor(end/2)+1:end, :), 'all');
            
            % Descriptors detallats
            desc_detall = [
                pct_red, pct_blue, pct_white, pct_yellow, pct_black, ...
                mean_red, mean_green, mean_blue, std_red, std_green, ...
                mean_hue, hue_std, mean_saturation, sat_std, ...
                circularity, stats_forma(1).Solidity, stats_forma(1).Extent, ...
                perimeter^2 / (4 * pi * area), ...
                fourier_detall, pct_edges, contrast, energy, homogeneity, ...
                symmetry_x, symmetry_y, texton1, texton2
            ];
        end
        
    catch ME
        fprintf('Error en extractDescriptors5Models: %s\n', ME.message);
    end
end