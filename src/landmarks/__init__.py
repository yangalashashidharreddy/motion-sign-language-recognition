"""
landmarks — MediaPipe-based hand landmark extraction pipeline.

Public API
----------
- :class:`~src.landmarks.hand_detector.HandDetector`
- :class:`~src.landmarks.landmark_extractor.LandmarkExtractor`
- :class:`~src.landmarks.landmark_extractor.HandLandmarks`
- :class:`~src.landmarks.landmark_sequence.LandmarkSequenceBuilder`
- :class:`~src.landmarks.landmark_sequence.LandmarkSequence`
- :mod:`~src.landmarks.landmark_utils`
- :class:`~src.landmarks.save_landmarks.LandmarkSaver`
- :mod:`~src.landmarks.visualize_landmarks`

Pipeline overview
-----------------
Video frames
    → HandDetector      (MediaPipe Hands inference)
    → LandmarkExtractor (structured (21, 3) arrays per hand)
    → LandmarkSequence  (padded/truncated (T, 2, 21, 3) per video)
    → LandmarkSaver     (.npy or .csv output)
    → visualize_landmarks (debug drawing utilities)
"""
