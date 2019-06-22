#!/usr/bin/env bash
#SBATCH -J submission
#SBATCH -N 1
#SBATCH -n 5
#SBATCH -A emmanuel
#SBATCH -p short
#SBATCH --mem 20G
#SBATCH --array=0-60%50
#SBATCH --output=slurm_messages/output/%A_%a.out
#SBATCH --error=slurm_messages/error/%A_%a.err

# Stop execution if there is an error
set -e

#Cleanup functions upon exit
function cleanup() {
	rm -rf $WORKDIR
}

trap cleanup EXIT SIGINT SIGTERM

# Directory setup
STARTDIR=$(pwd)
RAWDATADIR="$STARTDIR/ximpute_kalman/*.csv"
DATASETDIR="$STARTDIR/dataset/Extrasensory_uuid_fl_uTAR/"
DATAOUTDIR="$STARTDIR/changepoints/"
MYUSER=$(whoami)
LOCALDIR=/local

# Create an array of data file names
folder_directory=() # Empty array
for f in $RAWDATADIR  # Append data folder names to the empty array
do
	folder_directory+=($f)
done

# Directory setup continued
THISFILE="${folder_directory[$SLURM_ARRAY_TASK_ID]}"
echo ${THISFILE}
THISNAME="$(cut -d'/' -f6 <<<"$THISFILE")"
echo ${THISNAME}

THISUSER="$(cut -d'_' -f1 <<<"$THISNAME")"
echo ${THISUSER}

RAWDATAFILE="${DATASETDIR}${THISUSER}.features_labels.csv.gz"
THISJOB="c_$THISNAME"
WORKDIR=$LOCALDIR/$MYUSER/$THISJOB

rm -rf "$WORKDIR" && mkdir -p "$WORKDIR" && cd "$WORKDIR"

echo "Working on : "${folder_directory[$SLURM_ARRAY_TASK_ID]}
cp -a ${folder_directory[$SLURM_ARRAY_TASK_ID]} $WORKDIR
echo "Copied data csv to work dir."
cp -a ${RAWDATAFILE} $WORKDIR
echo "Copied tar file to work dir."
python3 $STARTDIR/changepoints_all.py
echo "Completed"
cp -a $WORKDIR/*.csv $DATAOUTDIR
echo "Copied output file"
#rm -rfv $WORKDIR/${folder_directory[$SLURM_ARRAY_TASK_ID]}
