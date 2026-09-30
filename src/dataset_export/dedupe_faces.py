"""Keep at most one face box per person box.

Used after a COCO dataset is written (YOLOX / RTMDet layout). The face
model can emit more than one box inside a single person crop - this
module assigns every face to the person it is most contained in, then
drops extras so each person keeps exactly one face (the largest / highest-
score one). Faces that do not sit inside any person box are left alone.
"""

import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import config


def _xywh_to_xyxy(bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    return float(x), float(y), float(x + w), float(y + h)


def _box_area(xmin: float, ymin: float, xmax: float, ymax: float) -> float:
    return max(0.0, xmax - xmin) * max(0.0, ymax - ymin)


def _intersection_area(a: Tuple[float, float, float, float],
                        b: Tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return _box_area(ix1, iy1, ix2, iy2)


def _face_belongs_to_person(face_xyxy: Tuple[float, float, float, float],
                             person_xyxy: Tuple[float, float, float, float],
                             min_containment: float) -> Tuple[bool, float]:
    """A face belongs to a person if its center is inside the person box
    OR at least `min_containment` of the face's own area overlaps the
    person (covers faces that slightly overflow a tight person box)."""
    face_area = _box_area(*face_xyxy)
    if face_area <= 0:
        return False, 0.0
    inter = _intersection_area(face_xyxy, person_xyxy)
    containment = inter / face_area
    cx = (face_xyxy[0] + face_xyxy[2]) * 0.5
    cy = (face_xyxy[1] + face_xyxy[3]) * 0.5
    center_inside = (person_xyxy[0] <= cx <= person_xyxy[2]
                     and person_xyxy[1] <= cy <= person_xyxy[3])
    return (center_inside or containment >= min_containment), containment


def _keep_score(face_ann: dict, containment: float) -> Tuple[float, float, float]:
    """Higher is better. Prefer detector score when present (in-memory
    exports), then larger face area, then higher containment."""
    score = float(face_ann.get("score", 0.0))
    area = float(face_ann.get("area", 0.0))
    if area <= 0:
        x, y, w, h = face_ann["bbox"]
        area = max(0.0, w) * max(0.0, h)
    return (score, area, containment)


def _category_ids(coco: dict, person_name: str, face_name: str) -> Tuple[int, int]:
    name_to_id = {c["name"]: c["id"] for c in coco.get("categories", [])}
    if person_name not in name_to_id or face_name not in name_to_id:
        raise ValueError(
            f"COCO categories {name_to_id} do not contain both "
            f"'{person_name}' and '{face_name}'."
        )
    return name_to_id[person_name], name_to_id[face_name]


def dedupe_faces_in_coco_dict(coco: dict,
                               person_name: str = None,
                               face_name: str = None,
                               min_containment: float = None) -> Tuple[dict, int]:
    """Rewrite `coco` in place: at most one face per person, per image.

    Returns (coco, n_faces_removed).
    """
    person_name = person_name or config.PERSON_CLASS_NAME
    face_name = face_name or config.FACE_CLASS_NAME
    min_containment = (
        config.FACE_IN_PERSON_MIN_CONTAINMENT if min_containment is None else min_containment
    )
    person_id, face_id = _category_ids(coco, person_name, face_name)

    by_image: Dict[int, List[dict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        by_image[ann["image_id"]].append(ann)

    kept: List[dict] = []
    removed = 0
    for anns in by_image.values():
        persons = [a for a in anns if a["category_id"] == person_id]
        faces = [a for a in anns if a["category_id"] == face_id]
        others = [a for a in anns if a["category_id"] not in (person_id, face_id)]
        kept.extend(persons)
        kept.extend(others)

        person_xyxys = [_xywh_to_xyxy(p["bbox"]) for p in persons]
        assigned: Dict[int, List[Tuple[dict, float]]] = defaultdict(list)
        unassigned: List[dict] = []

        for face in faces:
            face_xyxy = _xywh_to_xyxy(face["bbox"])
            best_i: Optional[int] = None
            best_containment = -1.0
            for i, person_xyxy in enumerate(person_xyxys):
                belongs, containment = _face_belongs_to_person(
                    face_xyxy, person_xyxy, min_containment
                )
                if belongs and containment > best_containment:
                    best_i = i
                    best_containment = containment
            if best_i is None:
                unassigned.append(face)
            else:
                assigned[best_i].append((face, best_containment))

        kept.extend(unassigned)
        for group in assigned.values():
            if len(group) == 1:
                kept.append(group[0][0])
                continue
            winner = max(group, key=lambda item: _keep_score(item[0], item[1]))[0]
            kept.append(winner)
            removed += len(group) - 1

    for new_id, ann in enumerate(kept, start=1):
        ann["id"] = new_id
    coco["annotations"] = kept
    return coco, removed


def dedupe_coco_json_file(json_path: str,
                           person_name: str = None,
                           face_name: str = None,
                           min_containment: float = None) -> int:
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    coco, removed = dedupe_faces_in_coco_dict(
        coco, person_name=person_name, face_name=face_name, min_containment=min_containment
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(coco, f)
    return removed


def dedupe_coco_dataset_dir(dataset_dir: str,
                             splits: Iterable[str] = ("train", "val"),
                             person_name: str = None,
                             face_name: str = None,
                             min_containment: float = None) -> int:
    """Rewrite instances_{split}2017.json under dataset_dir/annotations/."""
    total_removed = 0
    for split in splits:
        json_path = os.path.join(dataset_dir, "annotations", f"instances_{split}2017.json")
        if not os.path.isfile(json_path):
            print(f"[dedupe_faces] skip missing {json_path}")
            continue
        removed = dedupe_coco_json_file(
            json_path,
            person_name=person_name,
            face_name=face_name,
            min_containment=min_containment,
        )
        print(f"[dedupe_faces] {json_path}: removed {removed} extra face box(es)")
        total_removed += removed
    return total_removed
