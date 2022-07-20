### 07/20/22
Issues:
- The bad tracking when noise added might be just a feature to reflect the noise.
- However, I should work through the list of noise individually to test whether they make sense
Fixes:
- Motion.Drive track correctly without noise. When added noise, tracking is terrible. (Feature/Bug???)
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