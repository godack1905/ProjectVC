%% PRÀCTICA VC - RECONEIXEMENT DE SENYALS DE TRÀNSIT
% Processament d'imatges (s'ha inclos a extractDescriptors)


train_path = 'imatges_senyals/train';

% Definim la categoria i el nom de la imatge a processar
categoria = 'limit';
nombre_archivo = 'road805.png';


%% 0. Carreguem la imatge

img_path = fullfile(train_path, categoria, nombre_archivo);

% Verifiquem el path
if ~exist(img_path, 'file')
    fprintf('ERROR: No es troba la imatge: %s\n', img_path);
    return;
end

img = imread(img_path);
figure, imshow(img), title('Input image')


%% 1. PROCESAMENT DE COLORS

% Ibtenim la imatge en hsv i graylevel
img_hsv = rgb2hsv(img);
img_gray = rgb2gray(img);

figure, imshow(img_gray), title('Imatge en graylevels')

% Obtenim les components dels colors
H = img_hsv(:,:,1);
S = img_hsv(:,:,2);
V = img_hsv(:,:,3);

[rows, cols, ~] = size(img);

VAL = double(ones(rows, cols));
nomralized_hsv = cat(3, H, S, VAL);
rgb = hsv2rgb(nomralized_hsv);
figure, imshow(rgb), title('Normalized Image')

R = rgb(:,:,1);
G = rgb(:,:,2);
B = rgb(:,:,3);


%% 2. Detecció per color

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

figure, imshow(colorMask), title('Mascara de colors')


%% 3. Operacions morfologiques

ee = strel('disk', 1);
morphologicMask = imclose(colorMask, ee);
morphologicMask = imopen(morphologicMask, ee);

figure, imshow(morphologicMask), title('Mascara de operacions morfologiques sobre el color')


img_aux = img;
for canal = 1:3
    canal_img = img_aux(:,:,canal);
    canal_img(~morphologicMask) = 0;
    img_aux(:,:,canal) = canal_img;
end

img_gray_aux = rgb2gray(img_aux);

%% 4. Detecció d'edges

edges = edge(img_gray_aux, 'Canny', [0.1 0.2]);

figure, imshow(edges), title('Edges amb Canny')

% Combinar color i edges
combi = morphologicMask | edges;
combi = imclose(combi, strel('disk', 2));
combi = imfill(combi, 'holes');
combi = bwareaopen(combi, 150);

figure, imshow(combi), title('Color + Canny')


%% 5. Extracció de contorns

contorns = bwboundaries(combi);
stats = regionprops(combi, 'Area', 'BoundingBox', 'Solidity');


%% 6. Apliquem la mascara a l'imatge original

img_procesada = img;
for canal = 1:3
    canal_img = img_procesada(:,:,canal);
    canal_img(~combi) = 0;
    img_procesada(:,:,canal) = canal_img;
end

figure, imshow(img_procesada), title('Imatge procesada')


