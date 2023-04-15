# Uses half the vram, allows fitting 1500x1500+ frames into 16gigs, which the original full-precision RAFT can't do.
half = False

# Save human-readable flow images along with motion vectors. Check /{your output dir}/videoFrames/out_flo_fwd folder.
# TODO this should be a job param
save_flow_img = True