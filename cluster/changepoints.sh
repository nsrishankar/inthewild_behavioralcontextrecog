#!/usr/bin/env bash
#SBATCH -J kalman
#SBATCH -N 1
#SBATCH -n 15
#SBATCH -A emmanuel
#SBATCH -p emmanuel 
#SBATCH -p long
#SBATCH --mem 30G

set -e

#Cleanup functions upon exit
function cleanup() {
	rm -rf $WORKDIR
}

# Directory setup
STARTDIR=$(pwd)
DATAOUTDIR="$STARTDIR/changepoints_kalman/"
MYUSER=$(whoami)
LOCALDIR=/local
THISJOB="cpts_${SLURM_JOB_NAME}"
RAWDATADIR="$STARTDIR/ximpute_kalman/*"
WORKDIR=$LOCALDIR/$MYUSER/$THISJOB


rm -rf "$WORKDIR" && mkdir -p "$WORKDIR" && cd "$WORKDIR"

trap cleanup EXIT SIGINT SIGTERM

#cp -a $STARTDIR/convergence_file_test.py $WORKDIR

for f in $RAWDATADIR
do
	echo "Directory $f"
	echo * | wc
	cp -a $f $WORKDIR
	echo -e "\t Copied Data folder"
	# Call to execute program with parameters
		
	python3 $STARTDIR/cpts_blstm_smc_bocpd.py 
	echo "Completed"
	cp -a $WORKDIR/*.pkl $DATAOUTDIR
	#rm -rf $RAWDATDIR/$f
	rm -rfv $WORKDIR/*

done

# Copy data files to $DATAOUTDIR
#mkdir -p "$DATAOUTDIR"
#cp -a *.pkl $DATAOUTDIR

# Remove working directory
rm -rf "$WORKDIR"
