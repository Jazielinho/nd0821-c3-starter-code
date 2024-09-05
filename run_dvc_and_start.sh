#!/bin/bash

echo "Creando directorios temporales para DVC..."
mkdir -p $HOME/dvc-cache $HOME/dvc-state
echo "Directorios temporales creados."

echo "Configurando la caché de DVC para usar los directorios temporales..."
dvc config cache.dir $HOME/dvc-cache
dvc config state.dir $HOME/dvc-state
dvc config cache.type symlink
dvc config cache.shared group

echo "Ejecutando dvc pull para descargar los archivos en sus carpetas originales..."
dvc pull

# Comprobar si dvc pull fue exitoso
if [ $? -eq 0 ]; then
    echo "dvc pull completado exitosamente."
else
    echo "Error al ejecutar dvc pull."
    exit 1
fi

echo "Iniciando el servidor Uvicorn..."
uvicorn starter.main:app --host 0.0.0.0 --port 10000
