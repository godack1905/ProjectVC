%% PRÀCTICA VC - RECONEIXEMENT DE SENYALS DE TRÀNSIT
% Funció d'extracció de descriptors

function descriptors = extractDescriptors(img)
    % Extreu 25 descriptors 
    
    descriptors = zeros(1, 25);
    
    % Verificar que la imagen no esté vacía
    if isempty(img)
        return;
    end
    
    try
        % ---------------------------
        % 1. PROCESSAMENT DE COLORS
        % ---------------------------
        
        % Obtenim la imatge en hsv i graylevel
        img_hsv = rgb2hsv(img);
        img_gray = rgb2gray(img);
        
        % Obtenim les components dels colors
        H = img_hsv(:,:,1);
        S = img_hsv(:,:,2);
        V = img_hsv(:,:,3);
        
        % Normalitzem els colors
        [rows, cols, ~] = size(img);
        VAL = double(ones(rows, cols));
        normalized_hsv = cat(3, H, S, VAL);
        rgb_norm = hsv2rgb(normalized_hsv);
        
        R = rgb_norm(:,:,1);
        G = rgb_norm(:,:,2);
        B = rgb_norm(:,:,3);
        
        % ---------------------------
        % 2. DETECCIÓ DE COLORS
        % ---------------------------
        
        % Detecció de vermells
        red1 = (R > 0.5) & (G < 0.5) & (B < 0.5);
        red2 = (H > 0.95 | H < 0.05) & (S > 0.5) & (V > 0.4);
        
        % Detecció de blaus
        blue1 = (B > 0.5) & (R < 0.3) & (G < 0.5);
        blue2 = (H > 0.55 & H < 0.7) & (S > 0.4) & (V > 0.3);
        
        % Detecció de grocs i taronjas
        yellow1 = (R > 0.5) & (G > 0.5) & (B < 0.3);
        yellow2 = (H > 0.1 & H < 0.2) & (S > 0.4) & (V > 0.4);
        orange1 = (R > 0.8) & (G > 0.4) & (B < 0.3);
        orange2 = (H > 0.05 & H < 0.1) & (S > 0.4) & (V > 0.4);
        
        % Combinem els colors
        colorMask = red1 | red2 | blue1 | blue2 | yellow1 | yellow2 | orange1 | orange2;
        
        % -----------------------------
        % 3. OPERACIONS MORFOLÓGIQUES
        % -----------------------------
        
        ee = strel('disk', 1);
        morphMask = imclose(colorMask, ee);
        morphMask = imopen(morphMask, ee);
        
        img_masked = img;
        for canal = 1:3
            canal_img = img_masked(:,:,canal);
            canal_img(~morphMask) = 0;
            img_masked(:,:,canal) = canal_img;
        end
        
        img_gray_masked = rgb2gray(img_masked);
        
        % --------------------
        % 4. DETECCIÓ D'EDGES
        % --------------------
        
        edges = edge(img_gray_masked, 'Canny', [0.1 0.2]);
        
        % Combinar color i edges
        combi = morphMask | edges;
        combi = imclose(combi, strel('disk', 2));
        combi = imfill(combi, 'holes');
        combi = bwareaopen(combi, 150);
        
        % --------------------------
        % 5. DESCRIPTORS DE COLOR
        % --------------------------
        
        total_pixels = sum(combi(:));
        
        if total_pixels > 0
            pct_red = sum(red1(combi) | red2(combi)) / total_pixels;
            pct_blue = sum(blue1(combi) | blue2(combi)) / total_pixels;
            pct_yellow = sum(yellow1(combi) | yellow2(combi) | orange1(combi) | orange2(combi)) / total_pixels;
            
            mean_red = mean(R(combi));
            mean_green = mean(G(combi));
            mean_blue = mean(B(combi));
        else
            pct_red = 0; pct_blue = 0; pct_yellow = 0;
            mean_red = 0; mean_green = 0; mean_blue = 0;
        end
        
        % -------------------------
        % 6. DESCRIPTORS DE FORMA
        % -------------------------
        
        if total_pixels > 100
            stats = regionprops(combi, 'Area', 'Perimeter', 'Eccentricity', ...
                                'Solidity', 'Extent', 'BoundingBox', ...
                                'MajorAxisLength', 'MinorAxisLength');
            
            if ~isempty(stats)
                areas = [stats.Area];
                [~, idx] = max(areas);
                stats = stats(idx);
                
                area = stats.Area;
                perimeter = stats.Perimeter;
                eccentricity = stats.Eccentricity;
                solidity = stats.Solidity;
                extent = stats.Extent;
                
                if perimeter > 0
                    circularity = (4 * pi * area) / (perimeter^2);
                else
                    circularity = 0;
                end
                
                bbox = stats.BoundingBox;
                if bbox(4) > 0
                    aspect_ratio = bbox(3) / bbox(4);
                else
                    aspect_ratio = 0;
                end
                
                if stats.MinorAxisLength > 0
                    axis_ratio = stats.MajorAxisLength / stats.MinorAxisLength;
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
        % 7. DESCRIPTORS DE FOURIER
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
        % 8. DESCRIPTORS DE TEXTURA
        % ----------------------------
        
        if total_pixels > 0
            % Detecció d'edges interns (els de Canny)
            pct_edges = sum(edges(combi)) / total_pixels;
            
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
        % 9. AFEGIM TOTS ELS DESCRIPTORS
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