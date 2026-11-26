### Placement from the current grasp

Estimate the held object's displacement from the end effector using the current
images, calibrated depth and robot pose. Use several visible object-interior
samples; discard pixels on the background or uncertain boundaries. Express the
object and end-effector positions in the same frame. For a carry that preserves
the grasp orientation, choose the end-effector destination by subtracting the
measured held-object displacement from the desired object destination. Measure
again if the grasp or orientation changes. Maintain the holding gripper command
and inspect alignment during descent.

### Recovery from a fixed-orientation stall

When a translation stalls, check the current object-to-target alignment and
clearance using images and calibrated depth. If the object is aligned but descent
still stalls and the available pose-control interface supports it, try a bounded
motion that adjusts position and orientation together. Keep the holding gripper command
while carrying an object. Inspect the resulting motion before continuing. If the
adjustment produces no progress or approaches an obstacle, reassess the geometry;
a stall alone does not identify its physical cause.
