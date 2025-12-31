%% PRÀCTICA VC - RECONEIXEMENT DE SENYALS DE TRÀNSIT
% Funcio per calcular la octagonality (descriptor de forma)

function octagonality = computeOctagonality(mask)
    try
        if sum(mask(:)) < 100
            octagonality = 0;
            return;
        end
        
        stats = regionprops(mask, 'Area', 'Perimeter', 'BoundingBox');
        if isempty(stats)
            octagonality = 0;
            return;
        end
        
        area = stats.Area;
        perimeter = stats.Perimeter;
        bbox = stats.BoundingBox;
        
        if perimeter == 0 || area == 0 || bbox(3) == 0 || bbox(4) == 0
            octagonality = 0;
            return;
        end
        
        % 1. Circularitat (els octógons tenen menys circularitat)
        circularity = (4 * pi * area) / (perimeter^2);
        
        % 2. Relación de aspecto (STOP es casi cuadrado)
        aspect_ratio = bbox(3) / bbox(4);
        aspect_score = 1 - min(abs(1 - aspect_ratio), 1);  % 1 si es cuadrado
        
        % 3. Compacitad normalitzada
        compactness = perimeter^2 / area;
        compactness_norm = min(compactness / 1000, 1);
        
        % 4. Relació perímetre/área normalitzada
        perimeter_area_ratio = perimeter / sqrt(area);
        par_norm = min(perimeter_area_ratio / 50, 1);
        
        % Convinar métrique
        octagonality = (0.4 * (1 - circularity)) + ...
                       (0.3 * aspect_score) + ...
                       (0.2 * compactness_norm) + ...
                       (0.1 * par_norm);
        
        % Normalitzar
        octagonality = max(0, min(1, octagonality));
        
    catch
        octagonality = 0;
    end
end