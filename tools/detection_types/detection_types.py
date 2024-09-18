from enum import unique, Enum
from itertools import chain
from pathlib import Path
from typing import List, Optional, Tuple, Union
import tools.detection_types.detections_pb2 as pb

import numpy as np
from dataclasses import dataclass


@unique
class HandSide(Enum):
    LEFT = 0
    RIGHT = 1


@unique
class HandState(Enum):
    NO_CONTACT = 0
    SELF_CONTACT = 1
    ANOTHER_PERSON = 2
    PORTABLE_OBJECT = 3
    STATIONARY_OBJECT = 4


@dataclass
class IntCoordinate:
    x: int
    y: int

    def to_protobuf(self):
        coordinate = pb.IntCoordinate()
        coordinate.x = self.x
        coordinate.y = self.y
        assert coordinate.IsInitialized()
        return coordinate

    @staticmethod
    def from_protobuf(coordinate: pb.IntCoordinate) -> "IntCoordinate":
        return IntCoordinate(x=coordinate.x, y=coordinate.y,)

    def __add__(self, other: "IntCoordinate") -> "IntCoordinate":
        return IntCoordinate(x=self.x + other.x, y=self.y + other.y)

    def __mul__(self, scaler: float) -> "IntCoordinate":
        return IntCoordinate(x=round(self.x * scaler), y=round(self.y * scaler))

    def __iter__(self):
        yield from (self.x, self.y)

    def scale(self, width_factor: float = 1, height_factor: float = 1) -> None:
        self.x = round(self.x * width_factor)
        self.y = round(self.y * height_factor)


@dataclass
class FloatCoordinate:
    x: np.float32
    y: np.float32

    def to_protobuf(self):
        coordinate = pb.FloatCoordinate()
        coordinate.x = self.x
        coordinate.y = self.y
        assert coordinate.IsInitialized()
        return coordinate

    @staticmethod
    def from_protobuf(coordinate: pb.FloatCoordinate) -> "FloatCoordinate":
        return FloatCoordinate(x=coordinate.x, y=coordinate.y,)

    def __add__(self, other: "FloatCoordinate") -> "FloatCoordinate":
        return FloatCoordinate(x=self.x + other.x, y=self.y + other.y)

    def __mul__(self, scaler: float) -> "FloatCoordinate":
        return FloatCoordinate(x=self.x * scaler, y=self.y * scaler)

    def __iter__(self):
        yield from (self.x, self.y)

    def scale(self, width_factor: float = 1, height_factor: float = 1) -> None:
        self.x = self.x * width_factor
        self.y = self.y * height_factor


@dataclass
class BBox:
    top_left: IntCoordinate
    width: int
    height: int

    def to_protobuf(self):
        bbox = pb.BBox()
        bbox.width = self.width
        bbox.height = self.height
        bbox.top_left.MergeFrom(self.top_left.to_protobuf())
        assert bbox.IsInitialized()
        return bbox

    @staticmethod
    def from_protobuf(bbox: pb.BBox) -> "BBox":
        return BBox(
            top_left=IntCoordinate.from_protobuf(bbox.top_left),
            width=bbox.width,
            height=bbox.height,
        )

    @property
    def center(self) -> IntCoordinate:
        return self.top_left + IntCoordinate(
            x=round(self.width / 2), y=round(self.height / 2)
        )

    @property
    def coords(self) -> List[Tuple[int, int]]:
        return [
            (self.top_left.x, self.top_left.y),
            (self.top_left.x + self.width),
            (self.top_left.y + self.height),
        ]

    def scale(self, width_factor: float = 1, height_factor: float = 1) -> None:
        self.top_left.scale(width_factor=width_factor, height_factor=height_factor)
        self.width = round(self.width * width_factor)
        self.height = round(self.height * height_factor)


@dataclass
class OffsetVector:
    direction: FloatCoordinate
    magnitude: np.float32

    @staticmethod
    def from_detection(offset_vector: List[np.float32]) -> "OffsetVector":
        magnitude, x, y = offset_vector
        # The magnitude is scaled by 1000 during training to ensure the gradient
        # magnitude is at a similar level to that of the unit offset vector components,
        magnitude *= 1e3
        # A 0.1 weight factor is applied to the x, y components to act as a loss scalar
        x *= 10
        y *= 10
        return OffsetVector(direction=FloatCoordinate(x, y), magnitude=magnitude)

    def to_protobuf(self):
        vector = pb.OffsetVector()
        vector.magnitude = self.magnitude
        vector.position.MergeFrom(self.direction.to_protobuf())
        assert vector.IsInitialized()
        return vector

    @staticmethod
    def from_protobuf(vector: pb.OffsetVector) -> "OffsetVector":
        return OffsetVector(
            direction=FloatCoordinate.from_protobuf(vector.position),
            magnitude=vector.magnitude,
        )

    @property
    def x(self):
        return self.direction.x

    @property
    def y(self):
        return self.direction.y

    def scale(self, width_factor: float = 1, height_factor: float = 1) -> None:
        x = self.x * self.magnitude
        y = self.y * self.magnitude
        new_x = x * width_factor
        new_y = y * height_factor
        magnitude = np.sqrt(new_x ** 2 + new_y ** 2)
        unit_x = new_x / magnitude
        unit_y = new_y / magnitude

        self.direction.x = unit_x
        self.direction.y = unit_y
        self.magnitude = magnitude


@dataclass
class HandDetection:
    bbox: BBox
    score: np.float32
    state: HandState
    offset: OffsetVector
    side: HandSide

    @staticmethod
    def from_detection(detection: List[np.float32]) -> "HandDetection":
        # detection:
        # [
        #  0:top_left_x,
        #  1:top_left_y,
        #  2:bottom_right_x,
        #  3:bottom_right_y,
        #  4:score,
        #  5:hand_state,
        #  6:hand_vector_magnitude,
        #  7:hand_vector_x,
        #  8:hand_vector_y,
        #  9:hand_size
        # ]
        assert len(detection) == 10

        bbox = make_bbox(detection)
        score = detection[4]
        hand_state = int(detection[5])
        object_offset_vector = detection[6:9]
        hand_side = detection[-1]

        return HandDetection(
            bbox=bbox,
            score=score,
            state=HandState(hand_state),
            offset=OffsetVector.from_detection(object_offset_vector),
            side=HandSide(hand_side),
        )

    def to_protobuf(self):
        detection = pb.HandDetection()
        detection.bbox.MergeFrom(self.bbox.to_protobuf())
        detection.score = self.score
        detection.state = self.state.value
        detection.offset.MergeFrom(self.offset.to_protobuf())
        detection.side = self.side.value
        assert detection.IsInitialized()
        return detection

    @staticmethod
    def from_protobuf(detection: pb.HandDetection) -> "HandDetection":
        return HandDetection(
            bbox=BBox.from_protobuf(detection.bbox),
            score=detection.score,
            state=HandState(detection.state),
            offset=OffsetVector.from_protobuf(detection.offset),
            side=HandSide(detection.side),
        )

    def scale(self, width_factor: float = 1, height_factor: float = 1) -> None:
        self.bbox.scale(width_factor=width_factor, height_factor=height_factor)
        self.offset.scale(width_factor=width_factor, height_factor=height_factor)


@dataclass
class ObjectDetection:
    bbox: BBox
    score: np.float32

    @staticmethod
    def from_detection(detection: List[np.float32]) -> "ObjectDetection":
        # detection:
        # [
        #  0:top_left_x,
        #  1:top_left_y,
        #  2:bottom_right_x,
        #  3:bottom_right_y,
        #  4:score,
        #  5:hand_state,               # unused
        #  6:hand_vector_magnitude,    # unused
        #  7:hand_vector_x,            # unused
        #  8:hand_vector_y,            # unused
        #  9:hand_location             # unused
        # ]
        return ObjectDetection(bbox=make_bbox(detection), score=detection[4])

    def to_protobuf(self):
        detection = pb.ObjectDetection()
        detection.bbox.MergeFrom(self.bbox.to_protobuf())
        detection.score = self.score
        assert detection.IsInitialized()
        return detection

    @staticmethod
    def from_protobuf(detection: pb.ObjectDetection) -> "ObjectDetection":
        return ObjectDetection(
            bbox=BBox.from_protobuf(detection.bbox), score=detection.score
        )

    def scale(self, width_factor: float = 1, height_factor: float = 1) -> None:
        self.bbox.scale(width_factor=width_factor, height_factor=height_factor)


@dataclass
class Detections:
    video_id: str
    frame_number: int
    objects: List[ObjectDetection]
    hands: List[HandDetection]

    def to_protobuf(self):
        detections = pb.Detections()
        detections.video_id = self.video_id
        detections.frame_number = self.frame_number
        detections.hands.extend([hand.to_protobuf() for hand in self.hands])
        detections.objects.extend([object.to_protobuf() for object in self.objects])
        assert detections.IsInitialized()
        return detections

    @staticmethod
    def from_detections(
        video_id: str,
        frame_number: int,
        hand_detections: Optional[List[np.float32]],
        object_detections: Optional[List[np.float32]],
    ) -> "Detections":
        if hand_detections is None:
            hand_detections = []
        else:
            hand_detections = [HandDetection.from_detection(d) for d in hand_detections]

        if object_detections is None:
            object_detections = []
        else:
            object_detections = [
                ObjectDetection.from_detection(d) for d in object_detections
            ]
        detections = Detections(
            video_id=video_id,
            frame_number=frame_number,
            hands=hand_detections,
            objects=object_detections,
        )
        return detections

    @staticmethod
    def from_protobuf(detections: pb.Detections) -> "Detections":
        return Detections(
            video_id=detections.video_id,
            frame_number=detections.frame_number,
            hands=[HandDetection.from_protobuf(pb) for pb in detections.hands],
            objects=[ObjectDetection.from_protobuf(pb) for pb in detections.objects],
        )

    @staticmethod
    def from_protobuf_str(pb_str: bytes) -> "Detections":
        pb_detection = pb.Detections()
        pb_detection.MergeFromString(pb_str)
        return Detections.from_protobuf(pb_detection)

    def compute_hand_to_object_correspondence(
        self, object_threshold: float = 0,
    ) -> List[int]:
        object_indices = [
            i for i, obj in enumerate(self.objects) if obj.score >= object_threshold
        ]
        objects_centers = [
            tuple(obj.bbox.center)
            for obj in self.objects
            if obj.score >= object_threshold
        ]
        if len(object_indices) == 0 or len(self.hands) == 0:
            return []
        matching_object_idx = []  # matching list
        for hand_detection in self.hands:
            if hand_detection.state.value == HandState.NO_CONTACT.value:
                matching_object_idx.append(-1)
            else:  # hand is in-contact
                offset_center = (
                    hand_detection.bbox.center
                    + hand_detection.offset.direction *
                    hand_detection.offset.magnitude
                )
                offset_center = np.array(tuple(offset_center))
                distance_to_objects = np.sum(
                    (objects_centers - offset_center) ** 2, axis=-1
                )
                closest_object_index = np.argmin(distance_to_objects)
                matching_object_idx.append(object_indices[closest_object_index])

        return matching_object_idx

    def scale(self, width_factor: float = 1, height_factor: float = 1) -> None:
        for det in chain(self.hands, self.objects):
            det.scale(width_factor=width_factor, height_factor=height_factor)


def make_bbox(detection: List[np.float32]) -> BBox:
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = detection[:4]
    width = int(np.round(bottom_right_x - top_left_x))
    height = int(np.round(bottom_right_y - top_left_y))
    x = int(np.round(top_left_x))
    y = int(np.round(top_left_y))
    return BBox(IntCoordinate(x=x, y=y), width=width, height=height)


def load_detections(filename: Union[str, Path]) -> List[Detections]:
    import pickle

    with open(filename, "rb") as f:
        return [Detections.from_protobuf_str(s) for s in pickle.load(f)]


def save_detections(detections: List[Detections], filepath: Union[str, Path]) -> None:
    import pickle

    with open(filepath, "wb") as f:
        pickle.dump([d.to_protobuf().SerializeToString() for d in detections], f)
