# Classificació de senyals de trànsit (MATLAB)

Conjunt petit d'scripts MATLAB per extreure descriptors, entrenar i provar
un classificador de senyals de trànsit utilitzant les imatges a `imatges_senyals/`.

Ús ràpid:
- Obre MATLAB i executa `descriptorsTrain.m` per extreure descriptors i entrenar.
- Executa `testImage.m` per provar imatges amb el model entrenat.
- Models i dades d'entrenament: `trainedModel.mat`, `train_data_table.mat`.

Estructura rellevant:
- `imatges_senyals/` (carpetes `train/` i `test/` per classe)
- `extractDescriptors.m`, `descriptorsTrain.m`, `processament.m`, `testImage.m`