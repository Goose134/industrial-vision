from src.inference.types import Detection
from src.inference.ppe_visual_stabilizer import PersonBoxSmoother, PPEVisualSmoother


def d(name, bbox, track_id=None, conf=.75):
    return Detection(
        class_id={'person':0,'helmet':5,'vest':6}[name], class_name=name,
        confidence=conf,
        x1=bbox[0],y1=bbox[1],x2=bbox[2],y2=bbox[3],track_id=track_id
    )


def test_person_box_smoothes_same_track():
    obj=PersonBoxSmoother(alpha=.5)
    obj.next_frame()
    first=obj.update(d('person',(10,10,110,210),track_id=1))
    obj.next_frame()
    next_=obj.update(d('person',(20,20,120,220),track_id=1))
    assert (first.x1,first.y1)==(10,10)
    assert (next_.x1,next_.y1)==(15,15)


def test_no_person_track_id_returns_raw():
    obj=PersonBoxSmoother(alpha=.5)
    raw=d('person',(10,10,110,210))
    assert obj.update(raw) is raw


def test_ppe_cache_maintains_helmet_three_missing_frames_then_drops():
    filt=PPEVisualSmoother(alpha=.5,hold_frames=3)
    person=d('person',(100,100,300,500),track_id=7)
    helmet=d('helmet',(145,100,190,150))
    filt.next_frame()
    got,held=filt.update(person,person,'helmet',[helmet])
    assert got is not None and not held
    for i in range(3):
        filt.next_frame()
        got,held=filt.update(person,person,'helmet',[])
        assert got is not None and held
    filt.next_frame()
    got,held=filt.update(person,person,'helmet',[])
    assert got is None and not held


def test_cached_ppe_moves_with_person():
    filt=PPEVisualSmoother(alpha=.5,hold_frames=3)
    person=d('person',(100,100,300,500),track_id=7)
    helmet=d('helmet',(145,100,190,150))
    filt.next_frame()
    got,_=filt.update(person,person,'helmet',[helmet])
    assert got.x1==145
    moved=d('person',(150,100,350,500),track_id=7)
    filt.next_frame()
    got,held=filt.update(moved,moved,'helmet',[])
    assert held and got.x1==195


def test_new_track_no_old_equipment():
    filt=PPEVisualSmoother(alpha=.5,hold_frames=3)
    person=d('person',(100,100,300,500),track_id=7)
    filt.next_frame()
    filt.update(person,person,'helmet',[d('helmet',(145,100,190,150))])
    another=d('person',(100,100,300,500),track_id=8)
    got,held=filt.update(another,another,'helmet',[])
    assert got is None and not held


def test_ppe_smoothing_on_same_track():
    filt=PPEVisualSmoother(alpha=.5,hold_frames=3)
    person=d('person',(100,100,300,500),track_id=7)
    filt.next_frame()
    a,_=filt.update(person,person,'vest',[d('vest',(115,200,280,350))])
    filt.next_frame()
    b,_=filt.update(person,person,'vest',[d('vest',(125,200,290,350))])
    assert a.x1==115
    assert b.x1==120


def test_changed_location_resets_smoothing():
    filt=PPEVisualSmoother(alpha=.5,hold_frames=3)
    person=d('person',(0,0,500,500),track_id=7)
    filt.next_frame()
    filt.update(person,person,'helmet',[d('helmet',(0,0,30,30))])
    filt.next_frame()
    moved,_=filt.update(person,person,'helmet',[d('helmet',(300,300,340,340))])
    assert moved.x1==300
