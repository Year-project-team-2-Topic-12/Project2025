# !/bin/bash
# This script backs up the models directory by creating a tar.gz archive with a timestamp.
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="backups"
ARCHIVE_NAME="models_backup_$TIMESTAMP.tar.gz"
mkdir -p $BACKUP_DIR
tar -czf $BACKUP_DIR/$ARCHIVE_NAME ./models/*.pkl ./models/*.npz ./models/*.csv ./models/*.pickle
rm -f ./models/*.pkl ./models/*.csv ./models/*.pickle ./models/*.npz
echo "Backup created at $BACKUP_DIR/$ARCHIVE_NAME and original files removed."