#!/bin/bash


echo "Iniciando el servidor Uvicorn..."
uvicorn starter.main:app --host 0.0.0.0 --port 10000
