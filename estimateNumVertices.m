%% PRÀCTICA VC - RECONEIXEMENT DE SENYALS DE TRÀNSIT
% Funcio per calcular el numero estimat de vertexos

function num_vertices = estimateNumVertices(mask)
    % Estima el nombre de vertexos del contorno
    try
        contorn = bwboundaries(mask, 'noholes');
        if isempty(contorn)
            num_vertices = 0;
            return;
        end
        
        boundary = contorn{1};
        
        % Simplificar el contorn
        if size(boundary, 1) > 50
            step = ceil(size(boundary, 1) / 50);
            simplified = boundary(1:step:end, :);
            num_vertices = size(simplified, 1);
        else
            num_vertices = size(boundary, 1);
        end
        
        % Ajustar par a formes comuns
        if num_vertices >= 7 && num_vertices <= 9
            num_vertices = 8;  % Octógono
        elseif num_vertices >= 3 && num_vertices <= 5
            num_vertices = 3;  % Triángulo
        elseif num_vertices > 12
            num_vertices = 12; % Círculo (muchos vértices)
        end
        
        num_vertices = min(max(num_vertices, 0), 20);
        
    catch
        num_vertices = 0;
    end
end