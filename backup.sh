# !/bin/bash
# This script backs up the models directory by creating a tar.gz archive with a timestamp.
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="backups"
archives=()
# get arguments if need to backup everything or specific parts


ARCHIVE_NAME="models_backup_$TIMESTAMP.tar.gz"
mkdir -p $BACKUP_DIR
tar -czf $BACKUP_DIR/$ARCHIVE_NAME ./models/*
archives+=("$BACKUP_DIR/$ARCHIVE_NAME")
# rm -f ./models/*

ARCHIVE_NAME="results_backup_$TIMESTAMP.tar.gz"
mkdir -p $BACKUP_DIR
tar -czf $BACKUP_DIR/$ARCHIVE_NAME ./results/*
archives+=("$BACKUP_DIR/$ARCHIVE_NAME")
# rm -f ./results/*

ARCHIVE_NAME="data_backup_$TIMESTAMP.tar.gz"
mkdir -p $BACKUP_DIR
tar -czf $BACKUP_DIR/$ARCHIVE_NAME ./data/*
archives+=("$BACKUP_DIR/$ARCHIVE_NAME")
# rm -f ./data/*

echo "Backup completed. Created archives:"
printf '%s\n' "${archives[@]}"