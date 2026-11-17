import unittest
import numpy as np
from src.inference.vehicle_nms import vehicle_nms_mask


class FakeBoxes:
    def __init__(self, items):
        # Each row: x1,y1,x2,y2,confidence,class_id
        a = np.asarray(items, dtype=float).reshape(-1, 6)
        self.xyxy = a[:, :4]
        self.conf = a[:, 4]
        self.cls = a[:, 5]

    def __len__(self):
        return len(self.conf)


NAMES = {0: "person", 1: "car", 2: "truck", 3: "bus", 4: "motorcycle", 5: "helmet", 6: "vest", 7: "fire"}


class VehicleNMSTest(unittest.TestCase):
    def test_same_vehicle_detected_as_car_and_truck(self):
        b = FakeBoxes([[10, 10, 110, 90, .74, 1], [12, 12, 111, 91, .62, 2]])
        mask, dropped = vehicle_nms_mask(b, NAMES, .60)
        self.assertEqual(mask.tolist(), [True, False])
        self.assertEqual(dropped, [(1, 0)])

    def test_two_neighbouring_cars_survive(self):
        b = FakeBoxes([[10, 10, 110, 90, .74, 1], [75, 12, 175, 91, .62, 2]])
        mask, _ = vehicle_nms_mask(b, NAMES, .60)
        self.assertEqual(mask.tolist(), [True, True])

    def test_person_and_helmet_never_suppressed(self):
        b = FakeBoxes([[10, 10, 110, 110, .80, 0], [10, 10, 110, 110, .70, 5], [10, 10, 110, 110, .60, 1]])
        mask, _ = vehicle_nms_mask(b, NAMES, .60)
        self.assertEqual(mask.tolist(), [True, True, True])

    def test_vehicle_does_not_suppress_person(self):
        b = FakeBoxes([[0, 0, 100, 100, .80, 1], [0, 0, 100, 100, .70, 0], [0, 0, 100, 100, .60, 2]])
        mask, _ = vehicle_nms_mask(b, NAMES, .60)
        self.assertEqual(mask.tolist(), [True, True, False])

    def test_empty_boxes(self):
        self.assertEqual(vehicle_nms_mask(FakeBoxes([]), NAMES)[0].tolist(), [])

    def test_invalid_threshold(self):
        with self.assertRaises(ValueError):
            vehicle_nms_mask(FakeBoxes([]), NAMES, 1.5)


if __name__ == "__main__":
    unittest.main()
