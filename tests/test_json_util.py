import unittest
import json
from datetime import datetime, date
from decimal import Decimal

from src.common.json_util import to_json_array


class TestJsonUtil(unittest.TestCase):
    def test_to_json_array_basic(self):
        s = to_json_array([{"a": 1}, {"b": "x"}])
        obj = json.loads(s)
        self.assertIsInstance(obj, list)
        self.assertEqual(obj[0]["a"], 1)
        self.assertEqual(obj[1]["b"], "x")

    def test_to_json_array_tuple_and_set(self):
        s1 = to_json_array((1, 2, 3))
        self.assertEqual(json.loads(s1), [1, 2, 3])
        s2 = to_json_array({1, 2})
        self.assertIsInstance(json.loads(s2), list)

    def test_to_json_array_handles_datetime_and_decimal(self):
        s = to_json_array([
            {"ts": datetime(2024, 1, 1, 12, 0, 0)},
            {"amount": Decimal("1.25")},
            {"day": date(2024, 1, 2)},
        ])
        obj = json.loads(s)
        self.assertTrue(obj[0]["ts"].startswith("2024-01-01T12:00:00"))
        self.assertEqual(obj[1]["amount"], 1.25)
        self.assertTrue(obj[2]["day"].startswith("2024-01-02"))

    def test_to_json_array_invalid_input(self):
        with self.assertRaises(ValueError):
            to_json_array({"a": 1})


if __name__ == "__main__":
    unittest.main()