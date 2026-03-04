#!/bin/bash
# Script to schedule training on the cluster using sbatch
# It schedules the first training job and then schedules the next one to start after the first one finishes
# This is usually done after 4 hrs. To detect whether the training has started after submission it
# checks the directory where the slurm logs are saved for the presence of the log file created by the first training job.
# If the log file is not present it means the training has not started yet and it waits for 5 minutes before checking again.
# Once the log file is detected it schedules the next training job.

# PS: Please Alice ppl don't kill me for this script.

# Get the initial contents of the slurm log directory to compare later
LOG_DIR="/home/s4861264/detectron2/projects/LoTSS-GRG-detect/slurm_output/"
INITIAL_LOG_FILES=$(ls $LOG_DIR/*.out 2>/dev/null)
MAX_JOBS=5 # Maximum number of training jobs to schedule (including the first one)
CURRENT_JOBS=1 # Counter for the number of scheduled jobs (starting with 1 for the first job)

# Prompt the user to specify if they want to start a new training session or continue an existing one
while true; do
    read -p "Do you want to start a new training session (y/n)? " START_NEW_SESSION
    case "$START_NEW_SESSION" in
        [Yy] ) START_NEW_SESSION="y"; break;;
        [Nn] ) START_NEW_SESSION="n"; break;;
        * ) echo "Please enter y or n.";;
    esac
done

# Function to check the contents of the slurm log directory for the presence of the log file created by the first training job
check_for_new_log_file() {
    CURRENT_LOG_FILES=$(ls "$LOG_DIR"/*.out 2>/dev/null)
    NEW_LOG_FILES=$(comm -13 <(echo "$INITIAL_LOG_FILES" | sort) <(echo "$CURRENT_LOG_FILES" | sort))
    if [ -n "$NEW_LOG_FILES" ]; then
        echo "New log file detected: $NEW_LOG_FILES"
        INITIAL_LOG_FILES="$CURRENT_LOG_FILES" # Update the initial log files to include the new one for the next check
        return 0 # Log file detected
    else
        echo "No new log file detected yet."
        return 1 # No log file detected
    fi
}

schedule_next_training_job() {
    END_TIME=$(date +%s)
    ELAPSED_TIME=$((END_TIME - START_TIME))
    echo "----------------------------------------------"
    echo "Elapsed time since job #$CURRENT_JOBS started: $ELAPSED_TIME seconds."
    echo "Scheduling the next training job..."
    sbatch /home/s4861264/detectron2/projects/LoTSS-GRG-detect/scripts/train_run_continue.slurm
    echo "Next training job scheduled. Total jobs scheduled so far: $((CURRENT_JOBS + 1))"
    CURRENT_JOBS=$((CURRENT_JOBS + 1))
}

# Schedule the first training job
if [[ "$START_NEW_SESSION" == "y" ]]; then
    echo "Starting a new training session."
    echo "----------------------------------------------"
    sbatch /home/s4861264/detectron2/projects/LoTSS-GRG-detect/scripts/train_run.slurm
else
    echo "Continuing an existing training session..."
    echo "----------------------------------------------"
    sbatch /home/s4861264/detectron2/projects/LoTSS-GRG-detect/scripts/train_run_continue.slurm
fi

sleep 30 # Wait for 30 seconds to ensure the first job is scheduled and the log file is created

while [ $CURRENT_JOBS -lt $MAX_JOBS ]; do
    if check_for_new_log_file; then
        echo "Training job #$CURRENT_JOBS has started at $(date). Waiting 4 hours before scheduling the next one."
        echo "----------------------------------------------"
        START_TIME=$(date +%s)
        # Sleep for 4 hours before scheduling the next training job to ensure the first one has enough time to finish
        sleep 14460 # Sleep for 4 hours and 1 minute (14460 seconds)
        schedule_next_training_job
    else
        echo "Training job #$CURRENT_JOBS has not started yet. Will check again in 5 minutes."
        while ! check_for_new_log_file; do
            sleep 300 # Wait for 5 minutes before checking again
        done
        echo "Training job #$CURRENT_JOBS has started at $(date). Waiting 4 hours before scheduling the next one."
        echo "----------------------------------------------"
        START_TIME=$(date +%s)
        # Sleep for 4 hours before scheduling the next training job to ensure the first one has enough time to finish
        sleep 14460 # Sleep for 4 hours and 1 minute (14460 seconds)

        schedule_next_training_job
    fi
done

echo "Maximum number of training jobs ($MAX_JOBS) has been scheduled. Exiting."
