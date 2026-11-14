"""Versioned file formats of ESTEL (Part II, §9.4 and §11; task T2.5).

``estel.io`` owns the schema of the files ESTEL writes: ``_schema.py`` holds its version and the
writer and reader that ``STStatistics.save`` and ``STStatistics.load`` go through. It exposes no
public name in 1.0.0. In step 2 it also reads and writes the JSON formats of STL-Dev-MPI (Part
III, §18.3).
"""
