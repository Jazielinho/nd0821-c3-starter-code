#!/bin/bash

echo "Creando directorios temporales..."
mkdir -p /tmp/dvc-cache /tmp/dvc-tmp /tmp/dvc-state

echo "Directorios temporales creados."

echo "Configurando DVC para usar los directorios temporales..."
dvc cache dir /tmp/dvc-cache
export DVC_TMP_DIR=/tmp/dvc-tmp
export DVC_STATE_DIR=/tmp/dvc-state

echo "Ejecutando dvc pull..."
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
