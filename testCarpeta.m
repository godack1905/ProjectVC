%% PRÀCTICA VC - RECONEIXEMENT DE SENYALS DE TRÀNSIT
% Testing amb totes les imatges d'una carpeta

% Noms dels descriptors
descriptor_names = {
    'PctRed', 'PctBlue', 'PctYellow', ...
    'MeanRed', 'MeanGreen', 'MeanBlue', ...
    'Area', 'Perimeter', 'Eccentricity', ...
    'Solidity', 'Extent', 'Circularity', ...
    'AspectRatio', 'AxisRatio', 'Compactness', ...
    'Fourier1', 'Fourier2', 'Fourier3', 'Fourier4', 'Fourier5', ...
    'PctEdges', 'Contrast', 'Correlation', ...
    'Energy', 'Homogeneity'
};

% Cargar el model entrenat
load('trainedModel.mat');

% Definir la carpeta amb les imatges de test
test_folder = 'imatges_senyals\test\zona_cotxe';  % Canvia aquesta ruta si cal

% Obtenir totes les imatges .png de la carpeta
image_files = dir(fullfile(test_folder, '*.png'));

fprintf('S''han trobat %d imatges a la carpeta: %s\n\n', length(image_files), test_folder);

% Crear figura per mostrar els resultats
figure('Position', [100, 100, 1200, 800]);

% Array per guardar totes les prediccions
clases_predichas = cell(length(image_files), 1);

% Processar cada imatge
for i = 1:length(image_files)
    % Llegir la imatge
    img_path = fullfile(test_folder, image_files(i).name);
    img_test = imread(img_path);
    
    % Extreure descriptors
    desc_test = extractDescriptors(img_test);
    
    % Crear taula amb els descriptors
    tabla_test = array2table(desc_test, 'VariableNames', descriptor_names);
    
    % Fer la predicció
    classe = trainedModel.predictFcn(tabla_test);
    
    % Guardar la predicció
    clases_predichas{i} = char(classe);
    
    % Mostrar resultat en consola
    fprintf('Imatge %d/%d: %s -> Classe: %s\n', ...
        i, length(image_files), image_files(i).name, char(classe));
    
    % Mostrar imatge amb la predicció (màxim 20 imatges)
    if i <= 20
        subplot(4, 5, i);
        imshow(img_test);
        title(sprintf('%s\n%s', image_files(i).name, char(classe)), ...
            'Interpreter', 'none', 'FontSize', 8);
    end
end

fprintf('\n✓ Processat completat!\n');

%% RESUMEN DE PREDICCIONES
fprintf('\n════════════════════════════════════════\n');
fprintf('       RESUM DE PREDICCIONS\n');
fprintf('════════════════════════════════════════\n');

% Contar cuántas predicciones de cada clase
clases_unicas = unique(clases_predichas);
fprintf('\nTotal d''imatges processades: %d\n\n', length(image_files));

for i = 1:length(clases_unicas)
    clase_actual = clases_unicas{i};
    cantidad = sum(strcmp(clases_predichas, clase_actual));
    porcentaje = (cantidad / length(image_files)) * 100;
    fprintf('  %s: %d imatges (%.1f%%)\n', clase_actual, cantidad, porcentaje);
end

fprintf('\n════════════════════════════════════════\n');