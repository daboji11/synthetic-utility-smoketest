# School continuation prompt — paste into Codex

Continue this repo's RoboCasa π0.5 human-left LoRA setup. Read docs/pi05_h50_lora_handoff.md,
records/dp_final_comparison.json, configs/pi05/checkpoint_manifest.json, and actual pi05 verification
records if present. Read AGENTS.md if present on the new host.

User is on a school Mac controlling a Linux CUDA rental via SSH. Do not assume access to home
Windows/WSL, D: or old RunPod/conversation. Ask for GPU-host SSH address if unavailable.
Never store credentials in Git. Clone/pull this repo, do not rebuild it.

DECISION: retire ACT and DP; retain source/checkpoints/videos.
DP original2/20(seeds13,21), human-full-finetune0/20, identical20seeds.
Actual finetune used batch4,10kupdates,peakLR1e-5; later proposedbatch32 never trained.
Cause of degradation is not proven. Do not resume DP debugging/tuning.

ACTIVE: official RoboCasa π0.5 pretrain_human300 published75000 checkpoint; strictly LoRA only
on the fixed28 H50 human OpenDrawer LEFT demos. Exact prompt open the left drawer.
No26right demos,MG_30,DreamGen,mixedtraining,clustering,selection.
Keep native20Hz human/env; don't silently resample16Hz or scaleactions.
Threecams,224native model input,state16 reordered,action12.
All non-LoRA weights includingvision frozen; don't silently unfreeze heads/fullmodel.

Source robocasa-benchmark/openpi@5a6beda9ff99da30b4e1b59320f6a32971d7c397.
HF robocasa/robocasa365_checkpoints@c484448aba1a9b60a04c9b0ca117241518ea69f3,
prefix pi05_pretrain_human300/multitask_learning/75000.
JAX Orbax+nativeassets, not DP/Torch/generalpi05base.
Only12.44GBparams+assets download. Isolated officiallocked env; JAXCUDA check.
Mac is client; train on Linux GPU. Don't install into oldDPenv.

LOCAL VERIFICATION: isolatedJAXCUDA, pipcheck and4interfacecontracttests passed.
Officialreader verifiedH50state reorder, rawactionpreservation andall3cameramasks.
Only49,987,584LoRAparameters trainable; no training executed.
FullcheckpointGPU load OOM on8GB4060 withabout2.9GB alreadyoccupied; nopolicyforward/envstep.
Official weights fully downloaded at home, but remain outofGit. Run smoke_pi05_checkpoint.py
on cloudGPU BEFORE rollout/training. Do not describe the localfullforward as passed.

DATA NOT IN GIT: user must bring h50_pi05_handoff_data.tar+.tar.sha256 from homeD: toMac/cloud.
Checksum records/pi05_data_transfer.json.208MiB,28eps,84videos,5916frames,20Hz.
Verify/unpack under data/processed/OpenDrawer without overwriting originals.
If absent, state missingarchive and arrange identicalH50left source preparation if authorized;
don't pretend data arrived with Git.

ORDER:
1. Inspect GPUhost/files, isolated setup, JAXCUDA, simulatorassets.
2. Transfer/verify28human dataset+officialcheckpoint; preflight(noupdates/noenvsteps).
3. User launches vanillaπ0.5 one then3/20episodes, explicitstats and streamed3viewvideos.
4. User launches50updateLoRA speed/VRAM check before long paid training.
   4090candidate not certifiedthreecam batch. OfficialLoRA>22.5GB/full>70GB.
   Choose batch from measurement, not DP4/32 assumptions.
5. Prepare user command for chosenbatch, initial10ksteps/peakLR1e-5/seed0, humanonly.
   Save fullnativeparams+assets+config+metrics and2500/5000/7500/finalcheckpoints.
6. User launchesLoRA3 then20samefixedseeds; explicitstats.
7. Back up actual checkpoint/data before nonpersistent rental stop.

USER STARTS TRAINING AND ROLLOUTS. Don't launch them automatically.
Model-forward-only smoke is not rollout/training verification.
Consult actual savedrecords; don't assume preparation means allruntimegates passed.
No SOTA/28demo/LoRA success guarantee, no unsupported batchcausedDPfailure.
Commit/push reviewable code/smallrecords when requested, never data/weights/videos.
