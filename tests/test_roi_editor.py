"""Тесты формы ROI; запуск: python -m unittest discover -s tests -p test_roi_editor.py -v"""
import unittest

from src.gui.roi_editor import validate_polygon


class ROIPolygonTests(unittest.TestCase):
    def test_rectangle(self):
        a = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.8], [0.1, 0.8]]
        self.assertEqual(validate_polygon(a), a)

    def test_concave_polygon(self):
        points = [[0.1, 0.1], [0.8, 0.1], [0.4, 0.4], [0.8, 0.8], [0.1, 0.8]]
        self.assertEqual(len(validate_polygon(points)), 5)

    def test_three_points_valid(self):
        self.assertEqual(len(validate_polygon([[0, 0], [1, 0], [0, 1]])), 3)

    def test_too_few_vertices(self):
        with self.assertRaises(ValueError):
            validate_polygon([[0, 0], [1, 0]])

    def test_bow_tie(self):
        with self.assertRaises(ValueError):
            validate_polygon([[0.1, 0.1], [0.9, 0.9], [0.1, 0.9], [0.9, 0.1]])

    def test_closing_duplicate(self):
        with self.assertRaises(ValueError):
            validate_polygon([[0.1, 0.1], [0.8, 0.2], [0.7, 0.9], [0.1, 0.1]])

    def test_outside_frame(self):
        with self.assertRaises(ValueError):
            validate_polygon([[-0.1, 0], [1, 0], [0, 1]])

    def test_thin_polygon(self):
        with self.assertRaises(ValueError):
            validate_polygon([[0.1, 0.1], [0.9, 0.1], [0.9, 0.1001]])

    def test_nan_coordinate(self):
        with self.assertRaises(ValueError):
            validate_polygon([[0.1, 0.1], [float('nan'), 0.2], [0.6, 0.7]])


if __name__ == '__main__':
    unittest.main()
