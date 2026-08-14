"""
Data loading modules for inference and training datasets used in this repository.

This package provides dataset loaders and utilities for:
- LOCATA
- STARSS23
- STAIRS26
- AudibleLight Eigenmike32-5 DCASE-STARSS23
- EigenScape raw

Modules
-------
locata_loader
    PyTorch dataset classes for loading multi-channel audio and ground truth
    Direction of Arrival annotations from LOCATA
starss_loader
    PyTorch dataset classes for loading multi-channel audio and ground truth
    Direction of Arrival annotations
stairs26_loader
    Full multi-channel STAIRS26 recordings for inference
audiblelight_loader
    File-level CSM-pair training dataset for AudibleLight
eigenscape_loader
    File-level CSM-pair training dataset for EigenScape raw

See Also
--------
locata_loader.LocataAudioDataset : Full LOCATA audio-record loader with optional ground truth
locata_loader.LocataGroundTruthLoader : LOCATA frame-level DoA parser
starss_loader.StarssAudioDataset : Full STARSS audio-record loader with ground truth
starss_loader.StarssGroundTruthLoader : STARSS ground truth CSV parser
stairs26_loader.Stairs26AudioDataset : Full STAIRS26 audio-record loader
audiblelight_loader.AudibleLightCSMPairDataset : AudibleLight CSM-pair training dataset
eigenscape_loader.EigenscapeCSMPairDataset : EigenScape CSM-pair training dataset
"""
