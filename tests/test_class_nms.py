"""Run from project root: python -m pytest tests/test_class_nms.py -q"""

import numpy as np
import pytest

from src.inference.class_nms import class_nms_mask


class FakeBoxes:
    def __init__(self, coords, classes, scores):
        self.xyxy = np.asarray(coords, dtype=float).reshape((-1, 4))
        self.cls = np.asarray(classes, dtype=int)
        self.conf = np.asarray(scores, dtype=float)

    def __len__(self):
        return len(self.cls)


NAMES = {0: "person", 1: "car", 2: "truck", 3: "helmet", 4: "vest", 5: "fire", 6: "smoke", 7: "spill"}


def test_fire_two_same_class_overlapping_boxes_are_reduced():
    boxes = FakeBoxes([[10, 10, 100, 100], [12, 12, 98, 98]], [5, 5], [.84, .40])
    keep, dropped = class_nms_mask(boxes, NAMES, {"fire": .50})
    assert keep.tolist() == [True, False]
    assert dropped == [(1, 0)]


def test_fire_smoke_can_overlap_and_both_remain():
    boxes = FakeBoxes([[0, 0, 100, 100], [0, 0, 100, 100]], [5, 6], [.9, .8])
    keep, _ = class_nms_mask(boxes, NAMES, {"fire": .50, "smoke": .60})
    assert keep.tolist() == [True, True]


def test_person_helmet_vest_can_overlap_and_all_remain():
    boxes = FakeBoxes([[0, 0, 100, 100]] * 3, [0, 3, 4], [.9, .8, .7])
    keep, _ = class_nms_mask(boxes, NAMES, {"helmet": .65, "vest": .65})
    assert keep.tolist() == [True, True, True]


def test_spill_duplicates_are_reduced():
    boxes = FakeBoxes([[0, 0, 100, 100], [1, 1, 100, 100]], [7, 7], [.7, .6])
    keep, _ = class_nms_mask(boxes, NAMES, {"spill": .50})
    assert keep.tolist() == [True, False]


def test_different_non_matching_boxes_remain():
    boxes = FakeBoxes([[0, 0, 20, 20], [35, 35, 55, 55]], [5, 5], [.8, .7])
    keep, _ = class_nms_mask(boxes, NAMES, {"fire": .50})
    assert keep.tolist() == [True, True]


def test_car_truck_untouched_by_per_class_nms():
    boxes = FakeBoxes([[0, 0, 100, 100], [0, 0, 100, 100]], [1, 2], [.9, .8])
    keep, _ = class_nms_mask(boxes, NAMES, {"fire": .50})
    assert keep.tolist() == [True, True]


def test_too_small_nms_iou_suppresses_any_overlap_of_same_class():
    boxes = FakeBoxes([[0, 0, 100, 100], [99, 99, 150, 150]], [5, 5], [.8, .7])
    keep, _ = class_nms_mask(boxes, NAMES, {"fire": 0.0})
    assert keep.tolist() == [True, False]


def test_invalid_threshold_rejected():
    boxes = FakeBoxes([[0, 0, 100, 100]], [5], [.8])
    with pytest.raises(ValueError):
        class_nms_mask(boxes, NAMES, {"fire": -0.1})


def test_nested_fire_with_low_iou_removed():
    # small region is only 1% of large => IoU=0.01 (<0.05)
    boxes = FakeBoxes([[0,0,200,200],[80,80,100,100]],[5,5],[0.49,0.11])
    keep, removed = class_nms_mask(boxes,NAMES,{"fire":0.05})
    assert keep.tolist()==[True,False]
    assert removed == [(1,0)]

def test_real_small_fire_hotspot_survives_when_strong():
    boxes = FakeBoxes([[0,0,200,200],[80,80,100,100]],[5,5],[0.49,0.42])
    keep, _ = class_nms_mask(boxes,NAMES,{"fire":0.05})
    assert keep.tolist()==[True,True]

def test_smoke_nested_unaffected():
    boxes = FakeBoxes([[0,0,200,200],[80,80,100,100]],[6,6],[0.49,0.11])
    keep,_=class_nms_mask(boxes,NAMES,{"smoke":0.05})
    assert keep.tolist()==[True,True]

def test_partial_overlap_not_containment():
    boxes = FakeBoxes([[0,0,200,200],[190,190,250,250]],[5,5],[0.49,0.11])
    keep,_=class_nms_mask(boxes,NAMES,{"fire":0.05})
    assert keep.tolist()==[True,True]
