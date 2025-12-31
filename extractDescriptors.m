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
        
         mascara_final = procesarFinal(img);
        
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