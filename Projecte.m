%% PRÀCTICA VC - RECONEIXEMENT AUTOMÀTIC DE SENYALS DE TRÀFIC


clear; clc; close all;
train_path = 'imatges_senyals/train';

% Definim la categoria i el nom de la imatge a processar
categoria = 'limit';
nombre_archivo = '003_0038.png';


%% 1. Carreguem la imatge

img_path = fullfile(train_path, categoria, nombre_archivo);

% Verifiquem el path
if ~exist(img_path, 'file')
    fprintf('ERROR: No es troba la imatge: %s\n', img_path);
    return;
end

img = imread(img_path);
figure, imshow(img), title('Input image')


%% 2. PROCESAMIENTO COMPLETO DE LA IMAGEN

% Redimensionem la imatge si es molt gran
img_original = im2double(img);
if size(img_original,1) > 400 || size(img_original,2) > 400
    img_original = imresize(img_original, [300 NaN]);
    fprintf('Imagen redimensionada a: %dx%d\n', size(img_original,1), size(img_original,2));
end

figure, imshow(img_original), title('Imatge original (posible escalat)')

% Ibtenim la imatge en hsv i graylevel
img_hsv = rgb2hsv(img_original);
img_gray = rgb2gray(img_original);

figure, imshow(img_gray), title('Imatge en graylevels')

% Obtenim les components dels colors
R = img_original(:,:,1);
G = img_original(:,:,2);
B = img_original(:,:,3);
H = img_hsv(:,:,1);
S = img_hsv(:,:,2);
V = img_hsv(:,:,3);

figure, imshow(H), title('HUE')
figure, imshow(S), title('SAT')


%% 3. Detecció per color

% Detecció de vermells
red1 = (R > 0.6) & (G < 0.4) & (B < 0.4);
red2 = (H > 0.95 | H < 0.05) & (S > 0.5) & (V > 0.4);

% Detecció de blaus
blue1 = (B > 0.5) & (R < 0.3) & (G < 0.5);
blue2 = (H > 0.55 & H < 0.7) & (S > 0.4) & (V > 0.3);

% Combinem els colors
colorMask = red1 | red2 | blue1 | blue2;

figure, imshow(colorMask), title('Mascara de colors')


%% 4. Operacions morfologiques

ee = strel('disk', 3);
morphologicMask = imopen(colorMask, ee);
morphologicMask = imclose(morphologicMask, ee);
morphologicMask = imfill(morphologicMask, 'holes');
morphologicMask = bwareaopen(morphologicMask, 100);

figure, imshow(colorMask), title('Mascara de operacions morfologiques sobre el color')


%% 5. Detecció d'edges

edges = edge(img_gray, 'Canny', [0.1 0.2]);

figure, imshow(edges), title('Edges amb Canny')

% Combinar color i edges
combi = morphologicMask | edges;
combi = imclose(combi, strel('disk', 2));
combi = imfill(combi, 'holes');
combi = bwareaopen(combi, 150);

figure, imshow(colorMask), title('Color + Canny')


%% 6. Extracció de contorns i ROI

contorns = bwboundaries(combi);
stats = regionprops(combi, 'Area', 'BoundingBox', 'Solidity');

if ~isempty(stats)
    [~, idx] = max([stats.Area]);
    roi_bbox = stats(idx).BoundingBox;
else
    roi_bbox = [];
end


%% 7. Apliquem la mascara a l'imatge original

img_procesada = img_original;
for canal = 1:3
    canal_img = img_procesada(:,:,canal);
    canal_img(~combi) = 0;
    img_procesada(:,:,canal) = canal_img;
end

figure, imshow(img_procesada), title('Imatge procesada')



%% 8. Informacio de resultats

fprintf('Contorns detectats: %d\n', length(contorns));
if ~isempty(roi_bbox)
    fprintf('ROI: [x=%.1f, y=%.1f, ancho=%.1f, alto=%.1f]\n', ...
            roi_bbox(1), roi_bbox(2), roi_bbox(3), roi_bbox(4));
else
    fprintf('No ROI\n');
end
fprintf('Píxeles de señal: %d\n', sum(combi(:)));
fprintf('Píxeles de fondo: %d\n', sum(~combi(:)));

fprintf('\n=== PROCESAMIENTO COMPLETADO ===\n');
