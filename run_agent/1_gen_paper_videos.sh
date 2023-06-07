# For some reason, MineRL kills the program unpredictably when we instantiate a couple of environments.
# A simple solution is to run the run_agent in an infinite loop and have the python script only generate videos
# that are not already present in the output directory. Then, whenever this error happens, the python script will
# exit with a non-zero exit code, which will cause the bash script to restart the python script.
# When it finishes all videos, it should exit with a zero exit code, which will cause the bash script to exit.

COMMAND="python steve1/run_agent/run_agent.py \
    --in_model data/weights/vpt/2x.model \
    --in_weights data/weights/steve1/steve1.weights \
    --prior_weights data/weights/steve1/steve1_prior.pt \
    --text_cond_scale 6.0 \
    --visual_cond_scale 7.0 \
    --gameplay_length 3000 \
    --save_dirpath data/generated_videos/paper_prompts"

# Run the command and get its exit status
$COMMAND
EXIT_STATUS=$?

# Keep running the command until the exit status is 0 (generates all videos)
while [ $EXIT_STATUS -ne 0 ]; do
    echo
    echo "Encountered an error (likely internal MineRL error), restarting (will skip existing videos)..."
    echo "NOTE: If not MineRL error, then there might be a bug or the parameters might be wrong."
    sleep 10
    $COMMAND
    EXIT_STATUS=$?
done
echo "Finished generating all videos."