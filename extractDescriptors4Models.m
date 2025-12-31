%% PRÀCTICA VC - RECONEIXEMENT DE SENYALS DE TRÀNSIT
% Funcio d'extracció de descriptors

function [desc_forma, desc_color, desc_detall] = extractDescriptors4Models(img)
    
    desc_forma = zeros(1, 17);  % 17 descriptors de forma
    desc_color = zeros(1, 15);  % 15 descriptors de color
    desc_detall = zeros(1, 30); % 30 descriptors de detall
    
    % Verificar que la imatge no estigui buida
    if isempty(img) || ndims(img) < 3
        return;
    end
    
    try
        % --------------------------
        % 1. PROCESSAMENT INICIAL
        % --------------------------

        img = im2double(img);
        
        img_hsv = rgb2hsv(img);
        img_gray = rgb2gray(img);
        
        H = img_hsv(:,:,1);
        S = img_hsv(:,:,2);
        V = img_hsv(:,:,3);
        
        [rows, cols, ~] = size(img);
        
        % Normalització de color
        VAL = double(ones(rows, cols));
        normalized_hsv = cat(3, H, S, VAL);
        rgb_norm = hsv2rgb(normalized_hsv);
        
        R = rgb_norm(:,:,1);
        G = rgb_norm(:,:,2);
        B = rgb_norm(:,:,3);
        
        % Detecció de colors
        red1 = (R > 0.5) & (G < 0.5) & (B < 0.5);
        red2 = (H > 0.95 | H < 0.05) & (S > 0.4) & (V > 0.3);
        
        blue1 = (B > 0.4) & (R < 0.4) & (G < 0.5);
        blue2 = (H > 0.55 & H < 0.7) & (S > 0.3) & (V > 0.2);
        
        yellow1 = (R > 0.5) & (G > 0.45) & (B < 0.3);
        yellow2 = (H > 0.1 & H < 0.2) & (S > 0.3) & (V > 0.3);
        orange1 = (R > 0.7) & (G > 0.3) & (B < 0.3);
        orange2 = (H > 0.05 & H < 0.1) & (S > 0.3) & (V > 0.3);
        
        % Màscara de colors
        colorMask = red1 | red2 | blue1 | blue2 | yellow1 | yellow2 | orange1 | orange2;
        
        % Filtrar per una minima saturacio 
        saturation_filter = S > 0.2;
        colorMask = colorMask & saturation_filter;
        
        % Operacions morfológiques
        se = strel('disk', 2);
        morphMask = imclose(colorMask, se);
        morphMask = imopen(morphMask, strel('disk', 1));
        morphMask = imfill(morphMask, 'holes');
        morphMask = bwareaopen(morphMask, 100);
        
        % Aplicar la màscara obtinguda a la imatge
        img_masked = img;
        for canal = 1:3
            canal_img = img_masked(:,:,canal);
            canal_img(~morphMask) = 0;
            img_masked(:,:,canal) = canal_img;
        end
        
        img_gray_masked = rgb2gray(img_masked);
        
        % Detecció d'edges
        edges = edge(img_gray_masked, 'Canny', [0.05 0.15]);
        
        % Convinar color i edges
        combi = morphMask | edges;
        combi = imclose(combi, strel('disk', 2));
        combi = imfill(combi, 'holes');
        combi = bwareaopen(combi, 150);
        
        % ----------------------------
        % 2. DESCRIPTORS DE FORMA 
        % ----------------------------
        total_pixels = sum(combi(:));
        
        if total_pixels > 100
            try
                stats = regionprops(combi, 'Area', 'Perimeter', 'Eccentricity', ...
                                   'Solidity', 'Extent', 'BoundingBox', ...
                                   'MajorAxisLength', 'MinorAxisLength', ...
                                   'Centroid');
                
                if ~isempty(stats)
                    % Agafar la regio mes gran
                    areas = [stats.Area];
                    [~, idx] = max(areas);
                    stats = stats(idx);
                    
                    % Descriptores bàsicos
                    area = double(stats.Area);
                    perimeter = double(stats.Perimeter);
                    eccentricity = double(stats.Eccentricity);
                    solidity = double(stats.Solidity);
                    extent = double(stats.Extent);
                    
                    % Circularitat
                    if perimeter > 0
                        circularity = (4 * pi * area) / (perimeter^2);
                    else
                        circularity = 0;
                    end
                    
                    % Aspect ratio
                    bbox = stats.BoundingBox;
                    if bbox(4) > 0
                        aspect_ratio = bbox(3) / bbox(4);
                    else
                        aspect_ratio = 0;
                    end
                    
                    % Relació d'eixos
                    if stats.MinorAxisLength > 0
                        axis_ratio = double(stats.MajorAxisLength) / double(stats.MinorAxisLength);
                    else
                        axis_ratio = 0;
                    end
                    
                    % Compacitat
                    if area > 0
                        compactness = perimeter^2 / (4 * pi * area);
                    else
                        compactness = 0;
                    end
                    
                    % Factor de forma
                    if perimeter > 0
                        form_factor = (4 * pi * area) / (perimeter^2);
                    else
                        form_factor = 0;
                    end
                    
                    % Numero de vertexos estimat
                    num_vertices = estimateNumVertices(combi);
                    
                    % Descriptors de Fourier
                    fourier_desc = zeros(1, 3);
                    try
                        contorn = bwperim(combi);
                        [y, x] = find(contorn);
                        if length(x) > 50
                            centro = stats.Centroid;
                            s = (x - centro(1)) + 1i * (y - centro(2));
                            z = fft(s(:));
                            if abs(z(1)) > 0
                                z_norm = z / abs(z(1));
                                fourier_desc = abs(z_norm(2:4))';
                            end
                        end
                    catch
                        fourier_desc = zeros(1, 3);
                    end
                    
                    % Rectangularitat
                    if bbox(3) > 0 && bbox(4) > 0
                        rectangularity = area / (bbox(3) * bbox(4));
                    else
                        rectangularity = 0;
                    end
                    
                    % Diametre equivalent
                    if area > 0
                        equiv_diameter = sqrt(4 * area / pi);
                    else
                        equiv_diameter = 0;
                    end
                    
                    % Octagonalitat
                    octagonality = computeOctagonality(combi);
                    
                    % Descriptors de forma (17 en total)
                    desc_forma = [circularity, eccentricity, solidity, extent, ...
                                  aspect_ratio, axis_ratio, compactness, form_factor, ...
                                  num_vertices, fourier_desc(1), fourier_desc(2), ...
                                  fourier_desc(3), perimeter, area, rectangularity, ...
                                  equiv_diameter, octagonality];
                    
                    % Asegurar que no hi ha NaN o Inf
                    desc_forma(isnan(desc_forma)) = 0;
                    desc_forma(isinf(desc_forma)) = 0;
                end
            catch ME
                % Si n'hi ha error, ficar-hi zeros
                desc_forma = zeros(1, 17);
            end
        end
        
        % ----------------------------
        % 3. DESCRIPTORS DE COLOR
        % ----------------------------
        if total_pixels > 0
            try
                % Percentatges de color en la regió segmentada
                pct_red = sum(red1(combi) | red2(combi)) / total_pixels;
                pct_blue = sum(blue1(combi) | blue2(combi)) / total_pixels;
                
                % Detectar blanc (alta lluminositat, baixa saturació)
                white_mask = (V > 0.7) & (S < 0.3);
                pct_white = sum(white_mask(combi)) / total_pixels;
                
                % Detectar groc
                yellow_total = sum(yellow1(combi) | yellow2(combi) | orange1(combi) | orange2(combi));
                pct_yellow = yellow_total / total_pixels;
                
                % Estadístiques de color nomes en la regió segmentada
                R_region = R(combi);
                G_region = G(combi);
                B_region = B(combi);
                H_region = H(combi);
                S_region = S(combi);
                V_region = V(combi);
                
                if ~isempty(R_region)
                    mean_red = mean(R_region);
                    mean_green = mean(G_region);
                    mean_blue = mean(B_region);
                    mean_hue = mean(H_region);
                    mean_saturation = mean(S_region);
                    mean_value = mean(V_region);
                    
                    std_red = std(R_region);
                    std_blue = std(B_region);
                    
                    % Ratios de color
                    if mean_blue > 0
                        red_blue_ratio = mean_red / mean_blue;
                    else
                        red_blue_ratio = 0;
                    end
                    
                    if mean_green > 0
                        red_green_ratio = mean_red / mean_green;
                    else
                        red_green_ratio = 0;
                    end
                    
                    % Entropia de color en canal H (Hue)
                    hue_hist = histcounts(H_region, linspace(0, 1, 11));
                    hue_hist = hue_hist(hue_hist > 0);
                    hue_hist = hue_hist / sum(hue_hist);
                    if ~isempty(hue_hist)
                        color_entropy = -sum(hue_hist .* log2(hue_hist + eps));
                    else
                        color_entropy = 0;
                    end
                    
                    % Descriptors de color (15 en total)
                    desc_color = [pct_red, pct_blue, pct_white, pct_yellow, ...
                                 mean_red, mean_green, mean_blue, mean_hue, ...
                                 mean_saturation, mean_value, std_red, std_blue, ...
                                 red_blue_ratio, red_green_ratio, color_entropy];
                    
                    % Asegurar que no hi ha NaN
                    desc_color(isnan(desc_color)) = 0;
                    desc_color(isinf(desc_color)) = 0;
                end
            catch
                % Si n'hi ha algun error, ficar zeros
                desc_color = zeros(1, 15);
            end
        end
        
        % -------------------------------------------------
        % 4. DESCRIPTORS DETALLATS (para modelos 3-4)
        % -------------------------------------------------
        if total_pixels > 0
            try
                % Percentatge de negre
                black_mask = (V < 0.2);
                pct_black = sum(black_mask(combi)) / total_pixels;
                
                % Estadistiques adicionals
                std_green = std(G(combi));
                hue_std = std(H(combi));
                saturation_std = std(S(combi));
                
                % Percentages d'edges interns
                pct_edges = sum(edges(combi)) / total_pixels;
                
                % Textura GLCM
                contrast = 0; energy = 0; homogeneity = 0;
                if sum(combi(:)) > 50
                    try
                        region_gray = uint8(img_gray_masked * 255);
                        region_gray(~combi) = 0;
                        glcm = graycomatrix(region_gray, 'Offset', [0 1], 'Symmetric', true);
                        stats_glcm = graycoprops(glcm);
                        contrast = stats_glcm.Contrast;
                        energy = stats_glcm.Energy;
                        homogeneity = stats_glcm.Homogeneity;
                    catch
                        % Si falla GLCM, usar valors per defecte
                    end
                end
                
                % Simetria
                symmetry_x = computeSymmetrySimple(combi, 'horizontal');
                symmetry_y = computeSymmetrySimple(combi, 'vertical');
                
                % Textons
                texton1 = mean(double(edge(img_gray, 'sobel')));
                texton2 = std(double(img_gray(:)));
                
                % Fourier adicional per a detall
                fourier4 = 0;
                if exist('z_norm', 'var')
                    if length(z_norm) >= 5
                        fourier4 = abs(z_norm(5));
                    end
                end
                
                % Usar circularitat calculada avans o calcular-ne una nova
                if ~exist('circularity', 'var')
                    circularity = 0;
                end
                
                % Descriptors detallats (30 en total)
                desc_detall = [pct_red, pct_blue, pct_white, pct_yellow, pct_black, ...
                              mean_red, mean_green, mean_blue, std_red, std_green, ...
                              mean_hue, hue_std, mean_saturation, saturation_std, ...
                              circularity, solidity, extent, compactness, ...
                              fourier_desc(1), fourier_desc(2), fourier_desc(3), fourier4, ...
                              pct_edges, contrast, energy, homogeneity, ...
                              symmetry_x, symmetry_y, texton1, texton2];
                
                % Asegurar que no hi ha NaN y tamany correcte
                desc_detall(isnan(desc_detall)) = 0;
                desc_detall(isinf(desc_detall)) = 0;
                
                if length(desc_detall) > 30
                    desc_detall = desc_detall(1:30);
                elseif length(desc_detall) < 30
                    desc_detall = [desc_detall, zeros(1, 30 - length(desc_detall))];
                end
                
            catch
                % Si n'hi ha error, ficar zeros
                desc_detall = zeros(1, 30);
            end
        end
        
    catch ME
        fprintf('Error en extractDescriptors4Models: %s\n', ME.message);
        % Retornem zeros en cas d'error
        desc_forma = zeros(1, 17);
        desc_color = zeros(1, 15);
        desc_detall = zeros(1, 30);
    end
end