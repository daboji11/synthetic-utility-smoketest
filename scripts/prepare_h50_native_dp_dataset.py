#!/usr/bin/env python
"""Export exactly 28 original-20Hz H50 left demos to the official DP LeRobot/GR00T reader."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"data/processed/OpenDrawer/h50_human_left_20fps_3cam"
TARGET=ROOT/"data/processed/OpenDrawer/h50_human_left_native_dp_20fps_256"
CAMERAS=["robot0_agentview_left","robot0_agentview_right","robot0_eye_in_hand"]

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+"\n")

def export(source,target):
    import cv2
    if target.exists():
        raise FileExistsError(f"Preserving existing dataset: {target}")
    source_info=json.loads((source/"meta/info.json").read_text())
    provenance=json.loads((source/"provenance_manifest.json").read_text())
    eps=sorted(pq.read_table(source/"meta/episodes/chunk-000/file-000.parquet").to_pylist(),
               key=lambda row:row["episode_index"])
    if len(eps)!=28 or source_info["fps"]!=20 or provenance["source_instruction"]!="open the left drawer":
        raise ValueError("Expected the fixed 28 human left-drawer original-20Hz demos")
    table=pa.concat_tables([pq.read_table(p) for p in sorted((source/"data").glob("**/*.parquet"))])
    action=np.asarray(table["action"].to_pylist(),dtype=np.float32)
    state=np.asarray(table["observation.state"].to_pylist(),dtype=np.float32)
    if action.shape!=(5916,12) or state.shape!=(5916,16):
        raise ValueError("Unexpected source schema/length")
    target.mkdir(parents=True)
    (target/"meta").mkdir()
    features={key:source_info["features"][key] for key in
              ["action","observation.state","timestamp","frame_index","episode_index","index","task_index"]}
    for cam in CAMERAS:
        features[f"observation.images.{cam}_image"]={
            "dtype":"video","shape":[256,256,3],"names":["height","width","channel"],
            "info":{"video.fps":20,"video.channels":3,"video.codec":"h264","video.pix_fmt":"yuv420p"}}
    info={"codebase_version":"v2.1","fps":20,"total_episodes":28,"total_frames":5916,
          "total_tasks":1,"total_videos":84,"chunks_size":1000,"features":features,
          "data_path":"data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
          "video_path":"videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"}
    dump(target/"meta/info.json",info)
    # Source H50 vector order differs from the native PandaOmron storage template.
    # This metadata remaps slices; the vectors themselves remain byte-for-byte identical.
    state_slices={"end_effector_position_relative":(0,3),"end_effector_rotation_relative":(3,7),
                  "gripper_qpos":(7,9),"base_position":(9,12),"base_rotation":(12,16)}
    action_slices={"end_effector_position":(0,3),"end_effector_rotation":(3,6),"gripper_close":(6,7),
                   "base_motion":(7,11),"control_mode":(11,12)}
    def split_meta(slices,column):
        # Official template defaults absolute=True for all modalities; preserve that padding convention.
        return {key:{"original_key":column,"start":a,"end":b,"dtype":"float32","absolute":True}
                for key,(a,b) in slices.items()}
    modality={"state":split_meta(state_slices,"observation.state"),"action":split_meta(action_slices,"action"),
              "video":{cam:{"original_key":f"observation.images.{cam}_image"} for cam in CAMERAS},
              "annotation":{"human.task_description":{"original_key":"task_index"}}}
    dump(target/"meta/modality.json",modality)
    stats={}
    for key,values in [("observation.state",state),("action",action)]:
        stats[key]={name:value.tolist() for name,value in
                    [("min",values.min(0)),("max",values.max(0)),("mean",values.mean(0)),("std",values.std(0)),
                     ("q01",np.quantile(values,.01,axis=0)),("q99",np.quantile(values,.99,axis=0))]}
    dump(target/"meta/stats.json",stats)
    (target/"meta/tasks.jsonl").write_text(json.dumps({"task_index":0,"task":"open the left drawer"})+"\n")
    lengths=[]
    for ep in eps:
        idx=ep["episode_index"]
        length=ep["length"]
        subset=table.slice(ep["dataset_from_index"],length)
        for column in ["action","observation.state","timestamp","frame_index","episode_index","index","task_index"]:
            if column not in subset.column_names:
                raise ValueError(f"Missing original {column}")
        subset=subset.select(list(features)[:7])
        path=target/f"data/chunk-000/episode_{idx:06d}.parquet"
        path.parent.mkdir(parents=True,exist_ok=True)
        pq.write_table(subset,path)
        reloaded=pq.read_table(path)
        for col in ["action","observation.state","timestamp"]:
            if not reloaded[col].equals(subset[col]):
                raise ValueError(f"Changed source values: {idx}/{col}")
        source_provenance=provenance["episodes"][idx]
        if source_provenance["source_indices"]!=list(range(length)):
            raise ValueError("Unexpected resampling in source dataset")
        for cam in CAMERAS:
            key=f"observation.images.{cam}_image"
            source_video=source/source_info["video_path"].format(
                video_key=key,chunk_index=ep[f"videos/{key}/chunk_index"],file_index=ep[f"videos/{key}/file_index"])
            start=ep[f"videos/{key}/from_timestamp"]
            out=target/f"videos/chunk-000/{key}/episode_{idx:06d}.mp4"
            out.parent.mkdir(parents=True,exist_ok=True)
            subprocess.run(["ffmpeg","-v","error","-nostdin","-ss",str(start),"-i",str(source_video),
                            "-frames:v",str(length),"-vf","scale=256:256:flags=bilinear",
                            "-c:v","libx264","-threads","2","-crf","0","-preset","veryfast",
                            "-pix_fmt","yuv420p",str(out)],check=True)
            cap=cv2.VideoCapture(str(out))
            if int(cap.get(cv2.CAP_PROP_FRAME_COUNT))!=length or cap.get(cv2.CAP_PROP_FPS)!=20:
                raise ValueError(f"Video temporal mismatch: {out}")
            decoded=0
            while True:
                ok,frame=cap.read()
                if not ok:
                    break
                if frame.shape!=(256,256,3):
                    raise ValueError("Invalid decoded resolution")
                decoded+=1
            cap.release()
            if decoded!=length:
                raise ValueError(f"Incomplete decoded episode {idx}/{cam}")
        lengths.append({"episode_index":idx,"tasks":["open the left drawer"],"length":length})
        print(f"EXPORT episode={idx+1}/28 source_episode={source_provenance['source_episode_index']} frames={length} cameras=3 fps=20",flush=True)
    (target/"meta/episodes.jsonl").write_text("".join(json.dumps(ep)+"\n" for ep in lengths))
    manifest={**provenance,"target_dataset":str(target),"source_processed_dataset":str(source),
              "storage_format":"official DP LeRobot/GR00T episode parquet/video + modality.json",
              "image_transform":"128x128 -> 256x256 bilinear; no recovered image detail",
              "normalization":"raw vectors unchanged; runtime pretrained normalizer, not applied during export",
              "all_numeric_values_preserved":True,"all_84_videos_fully_decoded":True,
              "task":"OpenDrawer","instruction_distribution":{"open the left drawer":28},
              "drawer_side_distribution":{"left":28,"right":0},"total_frames":5916}
    dump(target/"provenance_manifest.json",manifest)
    print("DATASET_READY "+json.dumps({"path":str(target),"episodes":28,"frames":5916,"fps":20,
                                     "action_dim":12,"stored_state_dim":16,"native_proprio_dim":9,
                                     "camera_count":3,"resolution":[256,256]}),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=SOURCE)
    parser.add_argument("--output",type=Path,default=TARGET)
    args=parser.parse_args()
    export(args.source,args.output)
if __name__=="__main__":
    main()
