python steve1/run_agent/run_agent.py \
--in_model data/weights/vpt/2x.model \
--in_weights data/weights/steve1/steve1.weights \
--prior_weights data/weights/steve1/steve1_prior.pt \
--text_cond_scale 6.0 \
--visual_cond_scale 7.0 \
--gameplay_length 3000 \
--save_dirpath data/generated_videos/paper_prompts