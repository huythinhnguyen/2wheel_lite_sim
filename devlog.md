### 07/20/22
Issues:
- The noise generation code does not reflect the spec setting. --> Need fix.
Fixes:
- Removed all the noise-generator code from Motion.Drive --> work perfectly

### 07/19/22
Issues:
- Removed noise from from in Drive but the Drive class still did not track pose accurately.
- Linear velo is about 0.87 of actual. Angular velo is shifted toward the negatives
Fixed:
- rotation around ICC works well!
- Motion.State work wonderfully!

### 07/05/22
Issues:
- ToyController was too complicated for testing.
- There might be bug in Motion.update()
- The rotation around ICC might be wrong.