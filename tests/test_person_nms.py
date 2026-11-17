import numpy as np
from src.inference.class_nms import class_nms_mask

class FakeBoxes:
    def __init__(self, coords, classes, conf):
        self.xyxy=np.asarray(coords,dtype=float)
        self.cls=np.asarray(classes,dtype=int)
        self.conf=np.asarray(conf,dtype=float)
    def __len__(self): return len(self.cls)

NAMES={0:'person',5:'helmet',6:'vest'}

def test_double_person_removed_before_tracker():
    # Пример вложенных рамок одного рабочего со скриншота.
    boxes=FakeBoxes([[85,108,346,648],[109,144,309,636]], [0,0], [.59,.37])
    keep,dropped=class_nms_mask(boxes,NAMES,{'person':.55})
    assert keep.tolist()==[True,False]
    assert dropped==[(1,0)]

def test_person_does_not_suppress_helmet_or_vest():
    boxes=FakeBoxes([[0,0,300,500],[50,30,120,90],[70,110,230,320]], [0,5,6],[.9,.7,.8])
    keep,_=class_nms_mask(boxes,NAMES,{'person':.55})
    assert keep.tolist()==[True,True,True]

def test_two_distinct_people_remain():
    boxes=FakeBoxes([[0,0,100,200],[95,0,190,200]], [0,0],[.9,.8])
    keep,_=class_nms_mask(boxes,NAMES,{'person':.55})
    assert keep.tolist()==[True,True]
