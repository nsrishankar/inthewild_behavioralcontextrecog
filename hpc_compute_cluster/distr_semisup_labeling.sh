#!/usr/bin/env bash
#SBATCH -J distr_semisup
#SBATCH -N 1
#SBATCH -n 15
#SBATCH -A emmanuel
#SBATCH -p short
#SBATCH --mem 50G
#SBATCH --array=0-60%3
#SBATCH --output=slurm_messages/output/1NN%A_%a.out
#SBATCH --error=slurm_messages/error/1NN%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=nsrishankar@wpi.edu

# Stop execution if there is an error
set -e

#Cleanup functions upon exit
function cleanup() {
	rm -rf $WORKDIR
}

trap cleanup EXIT SIGINT SIGTERM

# Directory setup
STARTDIR=$(pwd)
DATASETDIR="$STARTDIR/dataset/Extrasensory_uuid_fl_uTAR/*.csv.gz"
DATAOUTDIR="$STARTDIR/semisupervisedlabeling_1NN/"
MYUSER=$(whoami)
LOCALDIR=/local

# Create an array of data file names
folder_directory=() # Empty array
for f in $DATASETDIR  # Append data folder names to the empty array
do
	folder_directory+=($f)
done


# Directory setup continued
THISFILE="${folder_directory[$SLURM_ARRAY_TASK_ID]}"
THISNAME="$(cut -d'/' -f7 <<<"$THISFILE")"
THISUSER="$(cut -d'.' -f1 <<<"$THISNAME")"
echo ${THISFILE}
echo ${THISNAME}
echo ${THISUSER}

# RAWDATAFILE="${DATASETDIR}${THISUSER}.features_labels.csv.gz"
THISJOB="ss_$THISUSER"
WORKDIR=$LOCALDIR/$MYUSER/$THISJOB

rm -rf "$WORKDIR" && mkdir -p "$WORKDIR" && cd "$WORKDIR"

echo "Working on : "${folder_directory[$SLURM_ARRAY_TASK_ID]}
cp -a ${folder_directory[$SLURM_ARRAY_TASK_ID]} $WORKDIR
echo "Copied tar file to work dir."
python3 $STARTDIR/semisup_labeling_1NN.py
echo "Completed"
cp -a $WORKDIR/*.pkl $DATAOUTDIR
echo "Copied output file"
rm -rfv $WORKDIR
echo "Removed work directory"
