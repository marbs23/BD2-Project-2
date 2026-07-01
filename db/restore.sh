#!/bin/bash
#
# restore.sh — restaura los datos en el PRIMER arranque del contenedor de Postgres.
#
# Postgres ejecuta los scripts de /docker-entrypoint-initdb.d en orden alfabético y
# solo cuando el volumen de datos está vacío. Aquí 01-schema.sql ya creó el esquema;
# este 02-restore.sh carga el dump data-only si el usuario lo colocó en db/seed/.
#
# El dump se distribuye aparte (pesa demasiado para git): se descarga del Drive y se
# coloca en db/seed/dump.sql.gz. Sin dump, la BD queda con el esquema vacío.
set -e

if [ -f /seed/dump.sql.gz ]; then
    echo ">> Restaurando datos desde /seed/dump.sql.gz ..."
    gunzip -c /seed/dump.sql.gz | psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"
    echo ">> Restauración completa."
else
    echo ">> Sin dump en /seed: la BD queda solo con el esquema (vacía)."
fi
