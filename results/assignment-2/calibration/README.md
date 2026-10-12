# Historical Task 4 calibration evidence

This calibration and independent stationary checkpoint are separate from the
final 262-frame evaluation. The radius used for the final run is 0.50 m.
The original pre-checkpoint component snapshot is retained as
historical-relative-measurements.json; its pending status describes that older
snapshot. static-registration-and-error.json contains the later corrected
16-pair native truth comparison. The raw independent capture is preserved;
its initial summary used a parser that did not recognize Jazzy text. The report
explains that correction and the limitations of 5 cm map registration.

Eight image-reviewed observations of three chairs are labeled. KF104 is
excluded, with a conditional identity sensitivity trial kept separate.
The nine labeled/reviewed frame PNGs, original 140-frame JSONL, 60-row depth
probe, trial rows and contact sheet are included. The old chair labels name
relative table-side identities; they do not assert final-run object IDs.

From the course repository, reproduce without ROS or database writes:

~~~bash
python3 results/assignment-2/calibration/reproduce_chair_study.py --recording results/assignment-2/calibration/keyframes.jsonl --out /tmp/assignment2-calibration-reproduced
~~~

The script needs NumPy and reads the pure study functions from the combined
semantic worker. Use a fresh output path. Do not mix these clock epochs with
the final database or interpret numerical transform consistency as accuracy.
